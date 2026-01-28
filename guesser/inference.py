"""
Quick, Draw! Model Inference Script
Load trained models and run predictions on example images
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.data import DataConfig, ALL_QUICKDRAW_CATEGORIES
from torchvision import transforms


def load_model(model_path, model_type, device):
    """Load a trained model from checkpoint"""
    # First load checkpoint to determine num_classes
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer num_classes from the checkpoint
    if 'head.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['head.weight'].shape[0]
    elif 'fc.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
    else:
        # Fallback: try to find the last linear layer
        for key in checkpoint['model_state_dict'].keys():
            if 'weight' in key and len(checkpoint['model_state_dict'][key].shape) == 2:
                num_classes = checkpoint['model_state_dict'][key].shape[0]
    
    print(f"Detected {num_classes} classes from checkpoint")
    
    model = create_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model from {model_path}")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Val Accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, num_classes


def load_example_image(dataset, idx=0):
    """Load an example image from the dataset"""
    image, label = dataset[idx]
    return image, label


def predict(model, image, device):
    """Run inference on a single image"""
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item()


def visualize_predictions(image, true_label, predictions, categories):
    """Visualize the image and model predictions"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Show the image
    ax = axes[0]
    img_np = image.squeeze().cpu().numpy()
    ax.imshow(img_np, cmap='gray')
    ax.set_title(f"True Label:\n{categories[true_label]}", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Show predictions for each model
    model_names = ['MLP', 'ResNet-18', 'ViT']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    for i, (model_name, (pred_idx, confidence)) in enumerate(zip(model_names, predictions)):
        ax = axes[i + 1]
        
        # Determine if prediction is correct
        is_correct = pred_idx == true_label
        color = '#2ecc71' if is_correct else '#e74c3c'
        
        # Display prediction
        ax.text(0.5, 0.6, f"{model_name}", 
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=colors[i], transform=ax.transAxes)
        
        ax.text(0.5, 0.4, f"{categories[pred_idx]}", 
                ha='center', va='center', fontsize=12, 
                transform=ax.transAxes, wrap=True)
        
        ax.text(0.5, 0.2, f"Confidence: {confidence*100:.1f}%", 
                ha='center', va='center', fontsize=10, 
                color=color, fontweight='bold', transform=ax.transAxes)
        
        # Add checkmark or X
        symbol = '✓' if is_correct else '✗'
        ax.text(0.5, 0.9, symbol, ha='center', va='center', 
                fontsize=40, color=color, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    # Configuration
    models_dir = Path("models")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Model paths
    model_configs = [
        ("mlp", models_dir / "mlp_best.pth"),
        ("resnet18", models_dir / "resnet18_best.pth"),
        ("vit", models_dir / "vit_best.pth"),
    ]
    
    # Check if model files exist
    missing_models = []
    for model_type, model_path in model_configs:
        if not model_path.exists():
            missing_models.append(str(model_path))
    
    if missing_models:
        print("Error: Missing model files:")
        for model_path in missing_models:
            print(f"  - {model_path}")
        print("\nPlease ensure trained models are in the 'models/' directory")
        return
    
    # Load dataset to get an example
    print("Loading Quick, Draw! dataset...")
    
    # First, load one model to determine num_classes
    first_model_path = model_configs[0][1]
    checkpoint = torch.load(first_model_path, map_location=device)
    if 'head.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['head.weight'].shape[0]
    elif 'fc.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['fc.weight'].shape[0]
    else:
        num_classes = 50  # Default fallback
    
    print(f"Models were trained with {num_classes} classes")
    
    # Load dataset with correct number of categories
    data_config = DataConfig(
        categories=ALL_QUICKDRAW_CATEGORIES[:num_classes],
        max_per_class=4,
        train_split=0.8,
        batch_size=1,
        num_workers=0
    )
    
    train_loader, val_loader = data_config.get_loaders()
    dataset = val_loader.dataset
    
    print(f"Dataset loaded: {len(data_config.categories)} categories\n")
    
    # Load all models
    print("=" * 60)
    print("Loading Models")
    print("=" * 60)
    models = {}
    for model_type, model_path in model_configs:
        model, _ = load_model(model_path, model_type, device)
        models[model_type] = model
    print()
    
    # Interactive inference loop
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)
    
    while True:
        # Get user input for which image to test
        print(f"\nDataset has {len(dataset)} validation images")
        user_input = input(f"Enter image index (0-{len(dataset)-1}) or 'q' to quit: ").strip()
        
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        
        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(dataset):
                print(f"Invalid index. Please enter a number between 0 and {len(dataset)-1}")
                continue
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit")
            continue
        
        # Load example image
        image, true_label = load_example_image(dataset, idx)
        print(f"\nTrue label: {data_config.categories[true_label]}")
        
        # Run predictions
        predictions = []
        for model_name, (model_type, _) in zip(['MLP', 'ResNet-18', 'ViT'], model_configs):
            pred_idx, confidence = predict(models[model_type], image, device)
            predictions.append((pred_idx, confidence))
            
            is_correct = "✓" if pred_idx == true_label else "✗"
            print(f"{model_name:12s} → {data_config.categories[pred_idx]:20s} "
                  f"(confidence: {confidence*100:5.1f}%) {is_correct}")
        
        # Visualize
        fig = visualize_predictions(image, true_label, predictions, data_config.categories)
        plt.savefig(f'guesser/prediction_{idx}.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: guesser/prediction_{idx}.png")
        plt.show()


if __name__ == "__main__":
    main()
