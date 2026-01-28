"""Configuration presets for Quick, Draw! classification experiments."""

# Quick test configuration (small subset for fast testing)
QUICK_TEST_CONFIG = {
    "model_types": ["mlp", "resnet18", "vit"],
    "num_classes": 20,  # Use 20 categories for quick testing
    "epochs": 10,
    "batch_size": 128,
    "lr": 3e-4,
    "dataset": "quickdraw",
    "save_dir": "./runs/quick_test",
    "max_per_class": 1000,  # Small dataset for fast iteration
}

# Full comparison configuration (all three models on larger dataset)
FULL_COMPARISON_CONFIG = {
    "model_types": ["mlp", "resnet18", "vit"],
    "num_classes": 50,  # 50 categories
    "epochs": 30,
    "batch_size": 128,
    "lr": 3e-4,
    "dataset": "quickdraw",
    "save_dir": "./runs/full_comparison",
    "max_per_class": 10000,
    "augment": True,
}

# MLP-specific configuration
MLP_CONFIG = {
    "model_types": ["mlp"],
    "num_classes": 345,  # All Quick, Draw! categories
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "dataset": "quickdraw",
    "save_dir": "./runs/mlp_full",
    "max_per_class": 10000,
    "model_kwargs": {
        "mlp": {
            "hidden_dims": (512, 256, 128),
            "dropout": 0.3
        }
    }
}

# ResNet-18 configuration
RESNET_CONFIG = {
    "model_types": ["resnet18"],
    "num_classes": 345,
    "epochs": 50,
    "batch_size": 128,
    "lr": 1e-3,
    "optimizer_type": "sgd",
    "dataset": "quickdraw",
    "save_dir": "./runs/resnet_full",
    "max_per_class": 10000,
    "augment": True,
}

# Vision Transformer (ViT) configuration
VIT_CONFIG = {
    "model_types": ["vit"],
    "num_classes": 345,
    "epochs": 50,
    "batch_size": 64,  # Smaller batch for ViT (memory intensive)
    "lr": 3e-4,
    "dataset": "quickdraw",
    "save_dir": "./runs/vit_full",
    "max_per_class": 10000,
    "augment": True,
    "model_kwargs": {
        "vit": {
            "dim": 256,
            "depth": 6,
            "heads": 8,
            "mlp_dim": 512,
            "dropout": 0.1,
            "emb_dropout": 0.1
        }
    }
}

# High-performance configuration (for final experiments)
HIGH_PERF_CONFIG = {
    "model_types": ["mlp", "resnet18", "vit"],
    "num_classes": 345,
    "epochs": 100,
    "batch_size": 128,
    "lr": 1e-3,
    "dataset": "quickdraw",
    "save_dir": "./runs/high_perf",
    "max_per_class": 50000,  # Use more samples per class
    "augment": True,
}
