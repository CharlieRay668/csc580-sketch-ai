"""Training utilities for classification experiments on Quick, Draw! dataset."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import json
from pathlib import Path


class TrainingConfig:
    """Configuration for training classifiers."""
    
    def __init__(
        self,
        epochs=50,
        lr=3e-4,
        optimizer_type='adam',
        weight_decay=0.0,
        device=None,
        save_dir="./runs",
        checkpoint_freq=10
    ):
        """
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            optimizer_type: Optimizer type ('adam' or 'sgd')
            weight_decay: Weight decay for regularization
            device: Device to train on (auto-detects if None)
            save_dir: Directory to save checkpoints and logs
            checkpoint_freq: Save checkpoint every N epochs
        """
        self.epochs = epochs
        self.lr = lr
        self.optimizer_type = optimizer_type.lower()
        self.weight_decay = weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.checkpoint_freq = checkpoint_freq
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch on classification task.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion (e.g., CrossEntropyLoss)
        device: Device to train on
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss criterion
        device: Device to evaluate on
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(val_loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, config, model_name="model"):
    """Complete training loop for classification.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: TrainingConfig instance
        model_name: Name for saving checkpoints and logs
    
    Returns:
        Dictionary with training history
    """
    device = config.device
    model.to(device)
    
    # Setup optimizer
    if config.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    
    # Loss criterion for classification
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }
    
    best_val_acc = 0.0
    
    print(f"\n=== Training {model_name} ===")
    print(f"Device: {device}")
    print(f"Optimizer: {config.optimizer_type}")
    print(f"Learning rate: {config.lr}")
    print(f"Epochs: {config.epochs}\n")
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch)
        
        print(f"Epoch {epoch}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = config.save_dir / f"{model_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_path)
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
        
        # Periodic checkpoint - only keep last checkpoint, delete previous ones
        if epoch % config.checkpoint_freq == 0:
            # Delete previous periodic checkpoint if it exists
            for old_checkpoint in config.save_dir.glob(f"{model_name}_epoch*.pth"):
                old_checkpoint.unlink()
            
            # Save new checkpoint
            checkpoint_path = config.save_dir / f"{model_name}_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Delete the last periodic checkpoint before saving final
    for old_checkpoint in config.save_dir.glob(f"{model_name}_epoch*.pth"):
        old_checkpoint.unlink()
    
    # Save final model
    final_path = config.save_dir / f"{model_name}_final.pth"
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, final_path)
    
    # Save training history
    history_path = config.save_dir / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training completed!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Final validation accuracy: {val_acc:.2f}%")
    print(f"  Models saved to: {config.save_dir}\n")
    
    return history


def load_checkpoint(model, checkpoint_path, device='cpu'):
    """Load a model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    return model
