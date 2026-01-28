"""Utility functions for visualization and analysis."""
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path


def count_parameters(model):
    """Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in megabytes
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def plot_training_curves(histories, save_path=None, title="Training Curves"):
    """Plot training and validation curves for multiple models.
    
    Args:
        histories: Dict of {model_name: history_dict}
        save_path: Optional path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    for name, history in histories.items():
        epochs = history['epochs']
        ax1.plot(epochs, history['train_loss'], label=f"{name} (train)", linestyle='--', alpha=0.7)
        ax1.plot(epochs, history['val_loss'], label=f"{name} (val)")
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    for name, history in histories.items():
        epochs = history['epochs']
        ax2.plot(epochs, history['train_acc'], label=f"{name} (train)", linestyle='--', alpha=0.7)
        ax2.plot(epochs, history['val_acc'], label=f"{name} (val)")
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def load_history(history_path):
    """Load training history from JSON file.
    
    Args:
        history_path: Path to history JSON file
    
    Returns:
        History dictionary
    """
    with open(history_path, 'r') as f:
        return json.load(f)


def compare_models(results_dict):
    """Print comparison table of model results.
    
    Args:
        results_dict: Dict of {model_name: results}
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<15} {'Parameters':<15} {'Best Val Acc':<15} {'Final Val Acc':<15}")
    print("-"*80)
    
    for model_name, results in results_dict.items():
        params = results.get('num_parameters', 'N/A')
        best_acc = results.get('best_val_acc', 'N/A')
        final_acc = results.get('final_val_acc', 'N/A')
        
        if isinstance(params, int):
            params_str = f"{params:,}"
        else:
            params_str = str(params)
        
        if isinstance(best_acc, (int, float)):
            best_acc_str = f"{best_acc:.2f}%"
        else:
            best_acc_str = str(best_acc)
        
        if isinstance(final_acc, (int, float)):
            final_acc_str = f"{final_acc:.2f}%"
        else:
            final_acc_str = str(final_acc)
        
        print(f"{model_name:<15} {params_str:<15} {best_acc_str:<15} {final_acc_str:<15}")
    
    print("="*80 + "\n")


def visualize_predictions(model, dataset, num_samples=16, device='cpu'):
    """Visualize model predictions on sample images.
    
    Args:
        model: Trained PyTorch model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to run inference on
    """
    model.eval()
    model.to(device)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            img, label = dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)
            
            output = model(img_tensor)
            _, pred = torch.max(output, 1)
            
            # Display image
            img_np = img.squeeze().cpu().numpy()
            ax.imshow(img_np, cmap='gray')
            ax.set_title(f"True: {label}\nPred: {pred.item()}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
