#!/usr/bin/env python3
"""
Policy Training Script for NRP Nautilus Cluster
Trains guessing policies in a competitive multi-agent setting
"""

import sys
import os
import argparse
from pathlib import Path

# Add workspace to path
sys.path.insert(0, '/workspace')

from src.train_policies import MultiAgentTrainer


def main():
    parser = argparse.ArgumentParser(description='Train guessing policies on cluster')
    parser.add_argument('--num-classes', type=int, default=50,
                       help='Number of classes (default: 50)')
    parser.add_argument('--samples-per-class', type=int, default=500,
                       help='Samples per class (default: 500)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--max-batches', type=int, default=0,
                       help='Max batches per epoch (0=all, default: 0)')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save policies every N epochs (default: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GUESSING POLICY TRAINING - NRP NAUTILUS")
    print("=" * 60)
    print(f"CoSamples per class: {args.samples_per_class}")
    print(f"  Total samples: {args.num_classes * args.samples_per_class}")
    print(f"  nfiguration:")
    print(f"  Classes: {args.num_classes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max batches/epoch: {args.max_batches if args.max_batches > 0 else 'all'}")
    print(f"  Device: {args.device}")
    print(f"  Save interval: every {args.save_interval} epochs")
    print("=" * 60)
    print()
    
    # Check that models exist
    model_dir = Path('/workspace/models')
    required_models = ['mlp_best.pth', 'resnet18_best.pth', 'vit_best.pth']
    
    print("Checking for pre-trained models...")
    for model_file in required_models:
        model_path = model_dir / model_file
        if not model_path.exists():
            print(f"  ✗ Missing: {model_file}")
            sys.exit(1)
        else:
            print(f"  ✓ Found: {model_file}")
    print()
    
    # Initialize trainer
    print("Initializing multi-agent trainer...")
    trainer = MultiAgentTrainer(
        num_classes=args.num_classes,
        device=args.device,
        h5_path=None,  # Use on-the-fly stroke detection
        samples_per_class=args.samples_per_class
    )
    print()
    
    # Train
    max_batches = args.max_batches if args.max_batches > 0 else float('inf')
    
    try:
        trainer.train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_batches=max_batches
        )
        
        # Final save
        print("\nSaving final policies...")
        trainer.save_policies("final")
        
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Results saved to: /workspace/models/policies/")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TRAINING FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
