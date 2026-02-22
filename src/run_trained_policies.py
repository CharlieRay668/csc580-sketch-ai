"""
Run inference with trained guessing policies.
Demonstrates progressive sketch reveal with all 9 trained policies.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import DataConfig, ALL_QUICKDRAW_CATEGORIES
from src.policies import ConfidenceThresholdPolicy, TimeBasedPolicy, LearnedPolicy
from src.stroke_detection import StrokeDetector


def load_model(model_path, model_type, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer num_classes from checkpoint
    if 'head.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['head.weight'].shape[0]
    elif 'fc.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
    elif 'mlp_head.1.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['mlp_head.1.weight'].shape[0]
    else:
        num_classes = 50
    
    model = create_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, num_classes


def load_policy(policy_path, policy_type, num_classes, device):
    """Load a trained policy from checkpoint."""
    if policy_type == 'confidence':
        policy = ConfidenceThresholdPolicy(threshold=0.9)
        state_dict = torch.load(policy_path, map_location=device)
        policy.threshold = state_dict['threshold']
        return policy
    
    elif policy_type == 'time':
        policy = TimeBasedPolicy(num_strokes=5)
        state_dict = torch.load(policy_path, map_location=device)
        policy.num_strokes = state_dict['num_strokes']
        return policy
    
    elif policy_type == 'learned':
        # Match the training configuration: include_confidence=True, include_time=False
        policy = LearnedPolicy(num_classes=num_classes, threshold=0.5, 
                               include_confidence=True, include_time=False)
        # The saved file contains the decision_layer's state_dict directly
        state_dict = torch.load(policy_path, map_location=device)
        policy.decision_layer.load_state_dict(state_dict)
        policy.decision_layer.to(device)
        return policy
    
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def simulate_progressive_reveal(image, strokes):
    """Simulate progressive stroke-by-stroke reveal.
    
    Args:
        image: Full image tensor [C, H, W] normalized to [-1, 1]
        strokes: List of stroke masks from detect_strokes (numpy arrays with values 0-255)
        
    Yields:
        (partial_image, num_strokes_seen)
    """
    canvas = torch.full_like(image, -1.0)  # Start with background value -1.0
    
    for i, stroke in enumerate(strokes):
        # Convert numpy stroke to torch tensor and normalize to [-1, 1]
        if isinstance(stroke, np.ndarray):
            stroke_tensor = torch.from_numpy(stroke).float()
            # Convert from [0, 255] to [-1, 1]
            stroke_tensor = (stroke_tensor / 127.5) - 1.0
            # Only update pixels where stroke exists (> 0 in original)
            mask = torch.from_numpy(stroke > 0)
            # Apply mask to the canvas (handle channel dimension)
            if canvas.dim() == 3:  # [C, H, W]
                canvas[0][mask] = stroke_tensor[mask]
            else:  # [H, W]
                canvas[mask] = stroke_tensor[mask]
        
        yield canvas.clone(), i + 1


def run_inference_with_policy(model, policy, image, strokes, categories, device):
    """Run inference with a guessing policy on progressive reveal.
    
    Returns:
        (guessed, stroke_number, prediction, confidence)
    """
    policy.reset()
    
    for partial_image, num_strokes_seen in simulate_progressive_reveal(image, strokes):
        # Ensure proper dimensions: [batch, channels, height, width]
        if partial_image.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            partial_image = partial_image.unsqueeze(0).unsqueeze(0)
        elif partial_image.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            partial_image = partial_image.unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            logits = model(partial_image.to(device))
        
        # Check if policy wants to guess
        should_guess, pred_idx, confidence = policy.should_guess(
            logits, 
            elapsed_time=num_strokes_seen,
            num_strokes_seen=num_strokes_seen
        )
        
        if should_guess:
            return True, num_strokes_seen, pred_idx, confidence
    
    # Never guessed
    return False, len(strokes), None, None


def visualize_all_policies(image, strokes, true_label, results, categories):
    """Create visualization showing when each policy guessed."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original image in top-left
    ax = axes[0, 0]
    img_np = image.squeeze().cpu().numpy()
    ax.imshow(img_np, cmap='gray')
    ax.set_title(f"Original\n{categories[true_label]}", fontweight='bold')
    ax.axis('off')
    
    model_names = ['MLP', 'ResNet-18', 'ViT']
    policy_types = ['Confidence', 'Time', 'Learned']
    
    # Plot each policy result
    for i, model_name in enumerate(model_names):
        for j, policy_type in enumerate(policy_types):
            ax = axes[i, j + 1]
            
            key = f"{model_name.lower().replace('-', '')}_{policy_type.lower()}"
            guessed, stroke_num, pred_idx, confidence = results[key]
            
            # Reconstruct image up to when policy guessed
            canvas = torch.full_like(image, -1.0)  # Start with background value -1.0
            for stroke in strokes[:stroke_num]:
                # Convert numpy stroke to torch tensor and normalize to [-1, 1]
                if isinstance(stroke, np.ndarray):
                    stroke_tensor = torch.from_numpy(stroke).float()
                    # Convert from [0, 255] to [-1, 1]
                    stroke_tensor = (stroke_tensor / 127.5) - 1.0
                    # Only update pixels where stroke exists (> 0 in original)
                    mask = torch.from_numpy(stroke > 0)
                    # Apply mask to the canvas (handle channel dimension)
                    if canvas.dim() == 3:  # [C, H, W]
                        canvas[0][mask] = stroke_tensor[mask]
                    else:  # [H, W]
                        canvas[mask] = stroke_tensor[mask]
            
            canvas_np = canvas.squeeze().cpu().numpy()
            ax.imshow(canvas_np, cmap='gray')
            
            # Format title
            if guessed and pred_idx is not None:
                is_correct = pred_idx == true_label
                symbol = '✓' if is_correct else '✗'
                color = 'green' if is_correct else 'red'
                
                title = f"{model_name} - {policy_type}\n"
                title += f"Stroke {stroke_num}/{len(strokes)} {symbol}\n"
                title += f"{categories[pred_idx]} ({confidence*100:.0f}%)"
            else:
                title = f"{model_name} - {policy_type}\n"
                title += f"Never guessed\n(saw all {stroke_num} strokes)"
                color = 'gray'
            
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Define models and policies to load
    models_dir = Path("models")
    policies_dir = Path("models/trained_policies")
    
    model_configs = [
        ("mlp", models_dir / "mlp_best.pth", "MLP"),
        ("resnet18", models_dir / "resnet18_best.pth", "ResNet-18"),
        ("vit", models_dir / "vit_best.pth", "ViT"),
    ]
    
    policy_types = ['confidence', 'time', 'learned']
    
    # Check if files exist
    print("Checking files...")
    for model_type, model_path, _ in model_configs:
        if not model_path.exists():
            print(f"❌ Missing: {model_path}")
            return
        print(f"✓ Found: {model_path}")
    
    for model_type, _, _ in model_configs:
        for policy_type in policy_types:
            policy_path = policies_dir / f"{model_type}_{policy_type}_final.pth"
            if not policy_path.exists():
                print(f"❌ Missing: {policy_path}")
                return
            print(f"✓ Found: {policy_path}")
    
    print("\n" + "="*70)
    print("Loading models and policies...")
    print("="*70)
    
    # Load models
    models = {}
    num_classes = None
    for model_type, model_path, model_name in model_configs:
        model, nc = load_model(model_path, model_type, device)
        models[model_type] = model
        num_classes = nc
        print(f"✓ Loaded {model_name} ({num_classes} classes)")
    
    # Load policies
    policies = {}
    for model_type, _, model_name in model_configs:
        for policy_type in policy_types:
            policy_path = policies_dir / f"{model_type}_{policy_type}_final.pth"
            policy = load_policy(policy_path, policy_type, num_classes, device)
            key = f"{model_type}_{policy_type}"
            policies[key] = policy
            print(f"✓ Loaded {model_name} - {policy_type.capitalize()} policy")
    
    # Load dataset
    print(f"\nLoading Quick, Draw! dataset ({num_classes} classes)...")
    data_config = DataConfig(
        categories=ALL_QUICKDRAW_CATEGORIES[:num_classes],
        max_per_class=10,
        train_split=0.8,
        batch_size=1,
        num_workers=0
    )
    
    _, val_loader = data_config.get_loaders()
    dataset = val_loader.dataset
    print(f"✓ Loaded {len(dataset)} validation images\n")
    
    # Interactive testing
    print("="*70)
    print("Running Progressive Reveal Inference")
    print("="*70)
    
    while True:
        user_input = input(f"\nEnter image index (0-{len(dataset)-1}) or 'q' to quit: ").strip()
        
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        
        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(dataset):
                print(f"Invalid index. Please enter 0-{len(dataset)-1}")
                continue
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")
            continue
        
        # Load image
        image, true_label = dataset[idx]
        print(f"\n{'='*70}")
        print(f"Image {idx}: {data_config.categories[true_label]}")
        print(f"{'='*70}")
        
        # Detect strokes
        print("Detecting strokes...")
        # Convert from [-1, 1] to [0, 255] properly
        img_np = ((image.squeeze().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
        detector = StrokeDetector()
        strokes = detector.detect_strokes(img_np)
        print(f"✓ Found {len(strokes)} strokes\n")
        
        if len(strokes) == 0:
            print("No strokes detected, skipping...\n")
            continue
        
        # Run inference with all policies
        results = {}
        print(f"{'Model':<15} {'Policy':<12} {'Guessed?':<10} {'Stroke':<12} {'Prediction':<20} {'Confidence'}")
        print("-" * 90)
        
        for model_type, _, model_name in model_configs:
            for policy_type in policy_types:
                key = f"{model_type}_{policy_type}"
                
                guessed, stroke_num, pred_idx, confidence = run_inference_with_policy(
                    models[model_type],
                    policies[key],
                    image,
                    strokes,
                    data_config.categories,
                    device
                )
                
                results[key] = (guessed, stroke_num, pred_idx, confidence)
                
                if guessed and pred_idx is not None:
                    is_correct = pred_idx == true_label
                    symbol = '✓' if is_correct else '✗'
                    pred_name = data_config.categories[pred_idx]
                    print(f"{model_name:<15} {policy_type.capitalize():<12} Yes {symbol:<7} "
                          f"{stroke_num}/{len(strokes):<9} {pred_name:<20} {confidence*100:.1f}%")
                else:
                    print(f"{model_name:<15} {policy_type.capitalize():<12} No        "
                          f"{stroke_num}/{len(strokes):<9} {'N/A':<20} N/A")
        
        # Visualize
        print(f"\nGenerating visualization...")
        fig = visualize_all_policies(image, strokes, true_label, results, data_config.categories)
        output_path = f'images/policy_comparison_{idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {output_path}\n")
        plt.close()


if __name__ == "__main__":
    main()
