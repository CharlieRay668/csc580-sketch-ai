"""Multi-agent policy training for Pictionary AI."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from src.policies import (
    GuessingPolicy, ConfidenceThresholdPolicy, TimeBasedPolicy, 
    LearnedPolicy, PolicyWrapper
)
from src.stroke_detection import StrokeDetector, ProgressiveStrokeRevealer
from src.utils import detect_num_classes
from src.data import QuickDrawDataset, ALL_QUICKDRAW_CATEGORIES
from src.models import create_model


class PreprocessedStrokeDataset(torch.utils.data.Dataset):
    """Dataset that loads preprocessed strokes from HDF5."""
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        self.labels = self.h5_file['labels'][:]
        self.num_strokes = self.h5_file['num_strokes'][:]
        self.length = len(self.labels)
        
        # Get image shape from first stroke
        if self.length > 0 and self.num_strokes[0] > 0:
            first_stroke = self.h5_file['strokes']['img_000000']['stroke_0']
            self.img_shape = tuple(first_stroke.attrs['shape'])
        else:
            self.img_shape = (28, 28)  # Default QuickDraw size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Returns (strokes, label) where strokes is a list of 2D numpy arrays."""
        label = self.labels[idx]
        num_strokes = self.num_strokes[idx]
        
        img_group = self.h5_file['strokes'][f'img_{idx:06d}']
        strokes = []
        
        for i in range(num_strokes):
            stroke_group = img_group[f'stroke_{i}']
            coords = stroke_group['coords'][:]
            values = stroke_group['values'][:]
            shape = tuple(stroke_group.attrs['shape'])
            
            # Reconstruct stroke as 2D array
            stroke_img = np.zeros(shape, dtype=np.uint8)
            stroke_img[coords[:, 0], coords[:, 1]] = values
            strokes.append(stroke_img)
        
        return strokes, label
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


class Agent:
    """Represents a model + policy combination."""
    
    def __init__(self, model_name: str, policy_name: str, policy: GuessingPolicy, 
                 wrapper: PolicyWrapper):
        self.model_name = model_name
        self.policy_name = policy_name
        self.policy = policy
        self.wrapper = wrapper
        self.name = f"{model_name}_{policy_name}"
        self.has_guessed = False
        self.guess_result = None
        self.decision_log_prob = None
        
    def reset(self):
        self.has_guessed = False
        self.guess_result = None
        self.decision_log_prob = None
        self.wrapper.reset()


class MultiAgentTrainer:
    """Trains guessing policies in a competitive multi-agent setting."""
    
    def __init__(self, num_classes: int = 50, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 h5_path: Optional[str] = None, samples_per_class: int = 5000):
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.h5_path = h5_path
        self.samples_per_class = samples_per_class
        self.stroke_detector = StrokeDetector() if h5_path is None else None
        self.revealer = ProgressiveStrokeRevealer()
        
        # Load the three trained models
        model_dir = Path('models')
        self.models = {}
        
        for model_name, model_type, model_file in [
            ('mlp', 'mlp', 'mlp_best.pth'), 
            ('resnet', 'resnet18', 'resnet18_best.pth'),
            ('vit', 'vit', 'vit_best.pth')
        ]:
            model_path = model_dir / model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Detect num_classes and create model
            num_classes_detected = detect_num_classes(str(model_path))
            model = create_model(model_type, num_classes=num_classes_detected)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device).eval()
            self.models[model_name] = model
        
        # Create agents (3 models × 3 policies = 9 agents)
        self.agents = self._create_agents()
        
        # Create optimizers for learned policies
        self.optimizers = {
            agent.name: optim.Adam(agent.policy.get_parameters(), lr=0.001)
            for agent in self.agents if agent.policy_name == 'learned'
        }
        
        print(f"Initialized {len(self.agents)} agents:")
        print(f"  - 3 LearnedPolicy (neural network)")
        print(f"  - 3 ConfidenceThresholdPolicy (adaptive threshold)")
        print(f"  - 3 TimeBasedPolicy (adaptive stroke count)")
    
    def _create_agents(self) -> List[Agent]:
        """Create all 9 agents."""
        agents = []
        
        for model_name, model in self.models.items():
            # Confidence policy - start more aggressive
            conf_policy = ConfidenceThresholdPolicy(threshold=0.70)
            conf_wrapper = PolicyWrapper(model, conf_policy, self.device)
            agents.append(Agent(model_name, 'confidence', conf_policy, conf_wrapper))
            
            # Time-based policy (stroke-based) - guess earlier
            time_policy = TimeBasedPolicy(num_strokes=3, use_best_guess=True)
            time_wrapper = PolicyWrapper(model, time_policy, self.device)
            agents.append(Agent(model_name, 'time', time_policy, time_wrapper))
            
            # Learned policy - lower threshold to encourage guessing
            learned_policy = LearnedPolicy(num_classes=self.num_classes, threshold=0.3,
                                          include_confidence=True, include_time=False)
            learned_policy.decision_layer.to(self.device)  # Move policy neural network to GPU
            learned_wrapper = PolicyWrapper(model, learned_policy, self.device)
            agents.append(Agent(model_name, 'learned', learned_policy, learned_wrapper))
        
        return agents
    
    def compute_rewards(self, agents: List[Agent], true_label: int) -> Dict[str, float]:
        """Compute rewards based on guessing performance.
        
        Rewards:
        - 1st correct: +10
        - 2nd correct: +5
        - 3rd correct: +2
        - 4th+ correct: +1
        - Wrong guess: -5
        - No guess: -10 (large penalty for never guessing)
        """
        rewards = {agent.name: -10.0 for agent in agents}  # Default penalty for not guessing
        
        # Separate correct and incorrect guesses
        correct_guesses = []
        incorrect_guesses = []
        
        for agent in agents:
            if agent.has_guessed:
                guess, conf, stroke_num = agent.guess_result
                if guess == true_label:
                    correct_guesses.append((agent, stroke_num))
                else:
                    incorrect_guesses.append(agent)
        
        # Sort correct guesses by stroke number (earlier = better)
        correct_guesses.sort(key=lambda x: x[1])
        
        # Assign rewards to correct guesses
        reward_schedule = [10.0, 5.0, 2.0, 1.0]
        for i, (agent, _) in enumerate(correct_guesses):
            reward = reward_schedule[i] if i < len(reward_schedule) else 1.0
            rewards[agent.name] = reward
        
        # Penalty for incorrect guesses
        for agent in incorrect_guesses:
            rewards[agent.name] = -5.0
        
        return rewards
    
    def train_episode(self, image_or_strokes, label: int, predetected: bool = False) -> Dict[str, float]:
        """Run one training episode with progressive stroke reveal.
        
        Args:
            image_or_strokes: Either a 2D numpy array (raw image) or list of 2D arrays (pre-detected strokes)
            label: Ground truth label
            predetected: If True, image_or_strokes is a list of pre-detected strokes
        """
        # Reset all agents
        for agent in self.agents:
            agent.reset()
        
        # Get ordered strokes
        if predetected:
            # Already detected, just order them
            ordered_strokes = image_or_strokes  # Assume already ordered from preprocessing
            if len(ordered_strokes) == 0:
                return {agent.name: 0.0 for agent in self.agents}
        else:
            # Detect strokes from raw image
            image = image_or_strokes
            strokes = self.stroke_detector.detect_strokes(image)
            if len(strokes) == 0:
                return {agent.name: 0.0 for agent in self.agents}
            ordered_strokes = self.stroke_detector.order_strokes_connected(strokes)
        
        # Generate progressive frames
        height, width = ordered_strokes[0].shape
        cumulative = np.zeros((height, width), dtype=np.uint8)
        frames = [cumulative.copy()]
        
        for stroke in ordered_strokes:
            mask = stroke > 0
            # For preprocessed strokes, grayscale values are already in the stroke
            cumulative[mask] = stroke[mask]
            frames.append(cumulative.copy())
        
        # Progressive reveal
        for stroke_idx in range(len(ordered_strokes)):
            current_frame = frames[stroke_idx + 1]  # +1 because frames[0] is empty
            
            # Convert to tensor [1, H, W]
            tensor_frame = torch.from_numpy(current_frame).float().unsqueeze(0) / 255.0
            
            # Query each agent
            for agent in self.agents:
                if agent.has_guessed:
                    continue
                
                # For learned policies, we need to track the decision probability
                if agent.policy_name == 'learned':
                    # Get model output
                    if tensor_frame.dim() == 3:
                        tensor_frame_batch = tensor_frame.unsqueeze(0)
                    else:
                        tensor_frame_batch = tensor_frame
                    
                    with torch.no_grad():
                        logits = agent.wrapper.model(tensor_frame_batch.to(self.device))
                    
                    # Get decision with gradient tracking
                    agent.policy.train_mode()
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    if probs.dim() > 1:
                        probs = probs[0]
                    
                    features = [probs]
                    if agent.policy.include_confidence:
                        features.append(torch.max(probs).unsqueeze(0))
                    
                    decision_logit = agent.policy.decision_layer(torch.cat(features))
                    decision_prob = torch.sigmoid(decision_logit)
                    
                    # Sample decision (guess or not)
                    should_guess = decision_prob.item() >= agent.policy.threshold
                    
                    if should_guess:
                        # Store log probability for policy gradient
                        agent.decision_log_prob = torch.log(decision_prob)
                        pred, conf = agent.policy._get_prediction(logits)
                        agent.has_guessed = True
                        agent.guess_result = (pred, conf, stroke_idx + 1)
                else:
                    # Fixed policies use normal inference
                    should_guess, pred, conf, logits = agent.wrapper.predict(
                        tensor_frame, num_strokes_seen=stroke_idx + 1
                    )
                    
                    if should_guess:
                        agent.has_guessed = True
                        agent.guess_result = (pred, conf, stroke_idx + 1)
            
            # Stop if all agents have guessed
            if all(agent.has_guessed for agent in self.agents):
                break
        
        # Compute rewards
        rewards = self.compute_rewards(self.agents, label)
        
        return rewards
    
    def update_policies(self, rewards: Dict[str, float]):
        """Update all policies based on performance."""
        # Determine ranking of agents
        agents_by_performance = []
        for agent in self.agents:
            if agent.has_guessed:
                guess, conf, stroke_num = agent.guess_result
                agents_by_performance.append((agent, rewards[agent.name], stroke_num))
        
        # Sort by reward (descending) then stroke number (ascending)
        agents_by_performance.sort(key=lambda x: (-x[1], x[2]))
        
        # Update each agent
        for rank, (agent, reward, stroke_num) in enumerate(agents_by_performance):
            was_correct = reward > 0  # Positive reward means correct
            was_late = rank >= 2  # Not in top 2
            
            if agent.policy_name == 'learned':
                # Neural network policy gradient update
                if agent.decision_log_prob is not None:
                    optimizer = self.optimizers[agent.name]
                    loss = -agent.decision_log_prob * reward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    agent.policy.eval_mode()
            
            elif agent.policy_name in ['confidence', 'time']:
                # Adaptive policy updates
                agent.policy.update(was_correct, was_late)
    
    def train(self, num_epochs: int = 10, batch_size: int = 32, max_batches: int = 50):
        """Train the policies over multiple epochs."""
        # Choose dataset based on whether we have preprocessed data
        if self.h5_path is not None:
            print(f"Using preprocessed strokes from: {self.h5_path}")
            dataset = PreprocessedStrokeDataset(self.h5_path)
            use_preprocessed = True
            
            # Simple collate function for preprocessed data
            def collate_fn(batch):
                strokes_list = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                return strokes_list, labels
        else:
            print("Using raw images with on-the-fly stroke detection")
            # Get categories
            categories = ALL_QUICKDRAW_CATEGORIES[:self.num_classes]
            
            # Create dataset (no transform, we need raw numpy arrays for stroke detection)
            dataset = QuickDrawDataset(
                root='data',
                categories=categories,
                max_per_class=self.samples_per_class,
                transform=None
            )
            use_preprocessed = False
            
            # Custom collate function to handle numpy arrays
            def collate_fn(batch):
                images = []
                labels = []
                for img, label in batch:
                    # Convert PIL Image to numpy if needed
                    if hasattr(img, 'numpy'):
                        img = img.numpy()
                    elif hasattr(img, 'convert'):
                        # PIL Image
                        img = np.array(img.convert('L'))
                    images.append(img)
                    labels.append(label)
                return images, labels
        
        # Create dataloader
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, collate_fn=collate_fn
        )
        
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Dataset: {len(dataset)} samples, {len(train_loader)} batches")
        
        for epoch in range(num_epochs):
            epoch_rewards = {agent.name: [] for agent in self.agents}
            
            # Training loop
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (data, labels) in enumerate(pbar):
                    # Process each item in the batch
                    for i in range(len(data)):
                        item = data[i]
                        label = labels[i]
                        
                        if use_preprocessed:
                            # item is already a list of strokes
                            strokes = item
                            rewards = self.train_episode(strokes, label, predetected=True)
                        else:
                            # item is a raw image, need to ensure it's a numpy array
                            image = item
                            if not isinstance(image, np.ndarray):
                                if hasattr(image, 'numpy'):
                                    image = image.numpy()
                                elif hasattr(image, 'convert'):
                                    image = np.array(image.convert('L'))
                            
                            # Ensure 2D array
                            if image.ndim == 3:
                                image = image.squeeze()
                            
                            rewards = self.train_episode(image, label, predetected=False)
                        
                        # Track rewards
                        for agent_name, reward in rewards.items():
                            epoch_rewards[agent_name].append(reward)
                        
                        # Update policies
                        self.update_policies(rewards)
                    
                    # Update progress bar with average rewards
                    if batch_idx % 10 == 0:
                        avg_rewards = {
                            name: np.mean(rlist) if rlist else 0.0 
                            for name, rlist in epoch_rewards.items()
                        }
                        pbar.set_postfix({
                            'mlp_learned': f"{avg_rewards['mlp_learned']:.2f}",
                            'resnet_learned': f"{avg_rewards['resnet_learned']:.2f}",
                            'vit_learned': f"{avg_rewards['vit_learned']:.2f}"
                        })
                    
                    # Limit batches for testing
                    if batch_idx >= max_batches - 1:
                        break
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print("Rewards:")
            for agent_name in sorted(epoch_rewards.keys()):
                avg_reward = np.mean(epoch_rewards[agent_name]) if epoch_rewards[agent_name] else 0.0
                print(f"  {agent_name:20s}: {avg_reward:6.2f}")
            
            # Print adaptive policy parameters
            print("\nAdaptive Policy Parameters:")
            for agent in self.agents:
                if agent.policy_name == 'confidence':
                    print(f"  {agent.name:20s} threshold: {agent.policy.threshold:.3f}")
                elif agent.policy_name == 'time':
                    print(f"  {agent.name:20s} num_strokes: {agent.policy.num_strokes}")
            
            # Save trainable policies
            if (epoch + 1) % 5 == 0:
                self.save_policies(f"epoch_{epoch+1}")
    
    def save_policies(self, suffix: str = "final"):
        """Save all policy parameters."""
        save_dir = Path('models/policies')
        save_dir.mkdir(exist_ok=True)
        
        for agent in self.agents:
            path = save_dir / f"{agent.name}_{suffix}.pth"
            
            if agent.policy_name == 'learned':
                # Save neural network weights
                torch.save(agent.policy.state_dict(), path)
            elif agent.policy_name == 'confidence':
                # Save threshold
                torch.save({'threshold': agent.policy.threshold}, path)
            elif agent.policy_name == 'time':
                # Save num_strokes
                torch.save({'num_strokes': agent.policy.num_strokes}, path)
        
        print(f"Saved all policies to {save_dir}/")


def main():
    trainer = MultiAgentTrainer(num_classes=50)
    trainer.train(num_epochs=10, batch_size=32, max_batches=50)


if __name__ == '__main__':
    main()
