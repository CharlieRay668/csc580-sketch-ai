#!/usr/bin/env python3
"""
Train all models (MLP, ResNet-18, ViT) on Quick, Draw! dataset.

This script provides global configuration variables for easy experimentation
and debugging. Set QUICK_TRAIN=True for fast debugging runs.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import create_model
from src.data import DataConfig, ALL_QUICKDRAW_CATEGORIES
from src.training import TrainingConfig, train_model
from src.utils import count_parameters, compare_models, plot_training_curves

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES - MODIFY THESE FOR YOUR EXPERIMENTS
# ============================================================================

# Quick train mode for debugging (uses minimal data and epochs)
QUICK_TRAIN = False  # Set to False for full training

# Models to train (comment out models you don't want to train)
MODELS_TO_TRAIN = [
    'mlp',
    'resnet18',
    'vit',
]

# Dataset Configuration
if QUICK_TRAIN:
    NUM_CATEGORIES = 10          # Small number of categories for quick testing
    MAX_SAMPLES_PER_CLASS = 500  # Few samples per class
    EPOCHS = 5                   # Very few epochs
    BATCH_SIZE = 64              # Smaller batch
else:
    NUM_CATEGORIES = 50          # More categories for real training
    MAX_SAMPLES_PER_CLASS = 10000  # Full dataset per class
    EPOCHS = 30                  # Full training
    BATCH_SIZE = 128             # Standard batch size

# Select specific categories (None = use first NUM_CATEGORIES from full list)
CUSTOM_CATEGORIES = None  # e.g., ['cat', 'dog', 'airplane', 'car', 'tree']

# Training Hyperparameters
LEARNING_RATE = 3e-4
OPTIMIZER_TYPE = 'adam'  # 'adam' or 'sgd'
WEIGHT_DECAY = 0.0
DATA_AUGMENTATION = not QUICK_TRAIN  # Enable augmentation for full training

# Paths
DATA_ROOT = "./data"
SAVE_DIR = "./runs/quickdraw_experiment"

# Device (auto-detects GPU if available)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Checkpoint frequency (save every N epochs)
CHECKPOINT_FREQ = 5 if not QUICK_TRAIN else 2

# Model-specific hyperparameters
MODEL_KWARGS = {
    'mlp': {
        'hidden_dims': (512, 256, 128) if not QUICK_TRAIN else (256, 128),
        'dropout': 0.3 if not QUICK_TRAIN else 0.2,
    },
    'resnet18': {
        # ResNet uses default parameters
    },
    'vit': {
        'dim': 256 if not QUICK_TRAIN else 128,
        'depth': 6 if not QUICK_TRAIN else 4,
        'heads': 8 if not QUICK_TRAIN else 4,
        'mlp_dim': 512 if not QUICK_TRAIN else 256,
        'dropout': 0.1,
        'emb_dropout': 0.1,
    }
}

# Logging
VERBOSE = True

# ============================================================================
# TRAINING SCRIPT - NO NEED TO MODIFY BELOW THIS LINE
# ============================================================================

def print_config():
    """Print current configuration."""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Mode: {'QUICK TRAIN (Debug)' if QUICK_TRAIN else 'FULL TRAINING'}")
    print(f"Models: {', '.join(MODELS_TO_TRAIN)}")
    print(f"Categories: {NUM_CATEGORIES}")
    print(f"Samples per class: {MAX_SAMPLES_PER_CLASS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Optimizer: {OPTIMIZER_TYPE}")
    print(f"Data augmentation: {DATA_AUGMENTATION}")
    print(f"Device: {DEVICE}")
    print(f"Save directory: {SAVE_DIR}")
    print("="*80 + "\n")


def main():
    """Main training function."""
    
    # Print configuration
    print_config()
    
    # Determine categories to use
    if CUSTOM_CATEGORIES is not None:
        categories = CUSTOM_CATEGORIES
        num_classes = len(categories)
        print(f"Using custom categories: {categories}\n")
    else:
        categories = ALL_QUICKDRAW_CATEGORIES[:NUM_CATEGORIES]
        num_classes = NUM_CATEGORIES
        print(f"Using first {num_classes} categories from Quick, Draw!\n")
    
    # Setup data configuration
    print("Setting up data loaders...")
    data_config = DataConfig(
        dataset="quickdraw",
        batch_size=BATCH_SIZE,
        data_root=DATA_ROOT,
        num_workers=4,
        categories=categories,
        max_per_class=MAX_SAMPLES_PER_CLASS,
        train_split=0.8,
        image_size=28,
        augment=DATA_AUGMENTATION
    )
    
    # Create data loaders
    train_loader, val_loader = data_config.get_loaders()
    
    # Setup training configuration
    training_config = TrainingConfig(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        optimizer_type=OPTIMIZER_TYPE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
        save_dir=SAVE_DIR,
        checkpoint_freq=CHECKPOINT_FREQ
    )
    
    # Store results for all models
    results = {}
    histories = {}
    
    # Train each model
    for model_type in MODELS_TO_TRAIN:
        print("\n" + "="*80)
        print(f"TRAINING {model_type.upper()}")
        print("="*80 + "\n")
        
        # Get model-specific kwargs
        model_kwargs = MODEL_KWARGS.get(model_type, {})
        
        # Create model
        print(f"Creating {model_type} model...")
        model = create_model(
            model_type,
            num_classes=num_classes,
            input_channels=1,  # Grayscale
            image_size=28,
            **model_kwargs
        )
        
        # Print model info
        num_params = count_parameters(model)
        print(f"Model parameters: {num_params:,}\n")
        
        # Train model
        try:
            history = train_model(
                model,
                train_loader,
                val_loader,
                training_config,
                model_name=model_type
            )
            
            # Store results
            results[model_type] = {
                'history': history,
                'num_parameters': num_params,
                'best_val_acc': max(history['val_acc']),
                'final_val_acc': history['val_acc'][-1]
            }
            histories[model_type] = history
            
            print(f"\n✓ {model_type.upper()} training completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error training {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80 + "\n")
    
    if results:
        compare_models(results)
        
        # Plot training curves if we have results
        if len(histories) > 0:
            try:
                print("Generating training curves plot...")
                plot_path = Path(SAVE_DIR) / "training_curves.png"
                plot_training_curves(
                    histories,
                    save_path=str(plot_path),
                    title=f"Quick, Draw! Classification ({num_classes} categories)"
                )
            except Exception as e:
                print(f"Could not generate plot: {e}")
    else:
        print("No models were successfully trained.")
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {SAVE_DIR}")
    print(f"- Model checkpoints: {SAVE_DIR}/*.pth")
    print(f"- Training histories: {SAVE_DIR}/*_history.json")
    if len(histories) > 0:
        print(f"- Training curves: {SAVE_DIR}/training_curves.png")
    print()


if __name__ == "__main__":
    main()
