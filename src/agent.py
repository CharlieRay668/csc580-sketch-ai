"""
PictionaryAgent: Unified interface for model inference with optional guessing policies.
Provides a clean API for both policy-based and non-policy model predictions.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image
import numpy as np
from torchvision import transforms

from src.models import create_model
from src.policies import (
    GuessingPolicy,
    ConfidenceThresholdPolicy,
    TimeBasedPolicy,
    LearnedPolicy
)
from src.stroke_detection import StrokeDetector

# The 50 Quick Draw categories used for training
QUICKDRAW_50_CATEGORIES = [
    'aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel',
    'animal migration', 'ant', 'anvil', 'apple', 'arm',
    'asparagus', 'axe', 'backpack', 'banana', 'bandage',
    'barn', 'baseball', 'baseball bat', 'basket', 'basketball',
    'bat', 'bathtub', 'beach', 'bear', 'beard',
    'bed', 'bee', 'belt', 'bench', 'bicycle',
    'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry',
    'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet',
    'brain', 'bread', 'bridge', 'broccoli', 'broom',
    'bucket', 'bulldozer', 'bus', 'bush', 'butterfly',
]


class PictionaryAgent:
    """
    An agent that combines a classification model with an optional guessing policy.
    
    Args:
        model_type: Type of model ('mlp', 'resnet18', 'vit')
        model_path: Path to trained model checkpoint (.pth file)
        policy_type: Type of policy ('confidence', 'time', 'learned', or None)
        policy_path: Path to trained policy checkpoint (required if policy_type is not None)
        categories: List of category names (defaults to first 50 Quick Draw categories)
        device: Torch device to use ('cuda' or 'cpu', auto-detected if None)
    """
    
    def __init__(
        self,
        model_type: str,
        model_path: str,
        policy_type: Optional[str] = None,
        policy_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        self.model_type = model_type
        self.policy_type = policy_type
        
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Load model
        self.model, self.num_classes = self._load_model(model_path)
        
        # Set categories
        self.categories = QUICKDRAW_50_CATEGORIES[:self.num_classes]
        
        # Load policy if specified
        self.policy = None
        self.has_guessed = False
        self.num_strokes_seen = 0
        
        if policy_type is not None:
            if policy_path is None:
                raise ValueError("policy_path is required when policy_type is specified")
            self.policy = self._load_policy(policy_path, policy_type)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    def _load_model(self, model_path: str) -> Tuple[torch.nn.Module, int]:
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Infer num_classes from checkpoint, should always be 50, but this is good practice
        if 'head.weight' in state_dict:
            num_classes = state_dict['head.weight'].shape[0]
        elif 'fc.weight' in state_dict:
            num_classes = state_dict['fc.weight'].shape[0]
        elif 'mlp_head.1.weight' in state_dict:
            num_classes = state_dict['mlp_head.1.weight'].shape[0]
        else:
            num_classes = 50
        
        model = create_model(self.model_type, num_classes=num_classes)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        return model, num_classes
    
    def _load_policy(self, policy_path: str, policy_type: str) -> GuessingPolicy:
        """Load a trained policy from checkpoint."""
        state_dict = torch.load(policy_path, map_location=self.device)
        
        if policy_type == 'confidence':
            policy = ConfidenceThresholdPolicy(threshold=0.9)
            policy.threshold = state_dict['threshold']
            return policy
        
        elif policy_type == 'time':
            policy = TimeBasedPolicy(num_strokes=5)
            policy.num_strokes = state_dict['num_strokes']
            return policy
        
        elif policy_type == 'learned':
            # Match training configuration: include_confidence=True, include_time=False
            policy = LearnedPolicy(
                num_classes=self.num_classes,
                threshold=0.5,
                include_confidence=True,
                include_time=False
            )
            policy.decision_layer.load_state_dict(state_dict)
            policy.decision_layer.to(self.device)
            return policy
        
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def reset(self):
        """Reset the agent state for a new drawing session."""
        self.has_guessed = False
        self.num_strokes_seen = 0
        if self.policy is not None:
            self.policy.reset()
    
    def predict(
        self,
        image: Image.Image,
        return_top_k: int = 5
    ) -> Dict:
        """
        Make a prediction on an image without policy (direct model inference).
        
        Args:
            image: PIL Image (can be RGBA or grayscale)
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with:
                - 'top_guess': str, the top prediction
                - 'top_confidence': float, confidence of top prediction
                - 'guesses': list of dicts with 'label' and 'confidence'
                - 'all_logits': tensor of all logits (for policy use)
        """
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Transform and predict
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            vals, idxs = torch.topk(probs, k=min(return_top_k, self.num_classes))
        
        guesses = [
            {"label": self.categories[i.item()], "confidence": v.item()}
            for v, i in zip(vals, idxs)
        ]
        
        return {
            "top_guess": guesses[0]["label"],
            "top_confidence": guesses[0]["confidence"],
            "guesses": guesses,
            "all_logits": logits
        }
    
    def predict_with_policy(
        self,
        image: Image.Image,
        num_strokes: Optional[int] = None,
        elapsed_time: Optional[float] = None,
        return_top_k: int = 5
    ) -> Dict:
        """
        Make a prediction with policy-based guessing.
        
        Args:
            image: PIL Image (can be RGBA or grayscale)
            num_strokes: Number of strokes seen so far (for time-based policy)
            elapsed_time: Elapsed time in seconds (for learned policy if configured)
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with:
                - 'should_guess': bool, whether policy decided to guess
                - 'has_guessed': bool, whether policy has already guessed (persistent)
                - 'top_guess': str or None, the top prediction (None if shouldn't guess)
                - 'top_confidence': float or None, confidence of top prediction
                - 'guesses': list of dicts with 'label' and 'confidence' (or None)
                - 'num_strokes_seen': int, current stroke count
        """
        if self.policy is None:
            raise ValueError("predict_with_policy requires a policy. Use predict() instead.")
        
        # Update stroke count if provided
        if num_strokes is not None:
            self.num_strokes_seen = num_strokes
        
        # If already guessed, return previous state
        if self.has_guessed:
            return {
                "should_guess": False,
                "has_guessed": True,
                "top_guess": None,
                "top_confidence": None,
                "guesses": None,
                "num_strokes_seen": self.num_strokes_seen
            }
        
        # Get prediction
        result = self.predict(image, return_top_k=return_top_k)
        logits = result['all_logits']
        
        # Check policy decision
        should_guess, pred_idx, confidence = self.policy.should_guess(
            logits,
            elapsed_time=elapsed_time,
            num_strokes_seen=self.num_strokes_seen
        )
        
        if should_guess:
            self.has_guessed = True
            return {
                "should_guess": True,
                "has_guessed": True,
                "top_guess": self.categories[pred_idx] if pred_idx is not None else result['top_guess'],
                "top_confidence": confidence if confidence is not None else result['top_confidence'],
                "guesses": result['guesses'],
                "num_strokes_seen": self.num_strokes_seen
            }
        else:
            return {
                "should_guess": False,
                "has_guessed": False,
                "top_guess": None,
                "top_confidence": None,
                "guesses": None,
                "num_strokes_seen": self.num_strokes_seen
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model and policy."""
        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "policy_type": self.policy_type,
            "has_policy": self.policy is not None,
            "device": str(self.device),
            "categories": self.categories
        }


def create_agent(
    model_name: str,
    models_dir: str = "models",
    policies_dir: str = "models/trained_policies",
    use_policy: bool = False,
    policy_type: Optional[str] = None
) -> PictionaryAgent:
    """
    Factory function to create a PictionaryAgent with common configurations.
    
    Args:
        model_name: 'mlp', 'resnet18', or 'vit'
        models_dir: Directory containing model checkpoints
        policies_dir: Directory containing policy checkpoints
        use_policy: Whether to load a policy
        policy_type: Type of policy ('confidence', 'time', 'learned')
        
    Returns:
        Configured PictionaryAgent instance
    """
    models_dir = Path(models_dir)
    policies_dir = Path(policies_dir)
    
    # Map friendly names to file names
    model_files = {
        'mlp': 'mlp_best.pth',
        'resnet18': 'resnet18_best.pth',
        'vit': 'vit_best.pth'
    }
    
    if model_name not in model_files:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model_path = models_dir / model_files[model_name]
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load policy if requested
    policy_path = None
    if use_policy:
        if policy_type is None:
            raise ValueError("policy_type must be specified when use_policy=True")
        
        policy_path = policies_dir / f"{model_name}_{policy_type}_final.pth"
        
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
    
    return PictionaryAgent(
        model_type=model_name,
        model_path=str(model_path),
        policy_type=policy_type if use_policy else None,
        policy_path=str(policy_path) if use_policy else None
    )
