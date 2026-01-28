# Quick, Draw! Classification Project

This codebase has been refactored from your masters thesis work to focus on training MLP, ResNet-18, and Vision Transformer (ViT) models on the Google Quick, Draw! dataset for your class project.

## 📁 Project Structure

```
src/
├── __init__.py           # Package initialization
├── models.py            # Model definitions (MLP, ResNet-18, ViT)
├── data.py              # Quick, Draw! dataset loader
├── training.py          # Training utilities and loops
├── experiments.py       # High-level experiment runners
├── configs.py           # Configuration presets
└── utils.py             # Visualization and analysis tools
```

## 🎨 Quick, Draw! Dataset

The Google Quick, Draw! dataset contains 50M hand-drawn sketches across 345 categories:
- 28×28 grayscale images
- Data automatically downloaded from Google Cloud Storage
- Configurable number of categories and samples per class

## 🧠 Models

### 1. MLP (Multi-Layer Perceptron)
- Simple fully-connected architecture
- Default: 3 hidden layers (512 → 256 → 128)
- Configurable dropout for regularization

### 2. ResNet-18
- Residual network with 18 layers
- Adapted for 28×28 images (no maxpool after first conv)
- 4 stages with [2, 2, 2, 2] basic blocks
- Channels: [64, 128, 256, 512]

### 3. ViT (Vision Transformer)
- Patch-based transformer architecture
- Default: 256-dim embeddings, 6 layers, 8 attention heads
- Configurable patch size (default: 4×4)
- Uses einops for efficient tensor operations

## 🚀 Quick Start

### Installation

```python
pip install torch torchvision tqdm einops matplotlib numpy pillow
```

### Basic Usage

```python
from src.models import create_model
from src.data import DataConfig
from src.training import TrainingConfig, train_model
from src.experiments import ExperimentConfig, run_classification_comparison

# Option 1: Quick test (20 categories, 10 epochs)
from src.configs import QUICK_TEST_CONFIG
config = ExperimentConfig(**QUICK_TEST_CONFIG)
results = run_classification_comparison(config)

# Option 2: Full comparison (all 3 models on 50 categories)
from src.configs import FULL_COMPARISON_CONFIG
config = ExperimentConfig(**FULL_COMPARISON_CONFIG)
results = run_classification_comparison(config)

# Option 3: Train individual model
data_config = DataConfig(
    dataset="quickdraw",
    batch_size=128,
    categories=None,  # Use default 20 categories
    max_per_class=5000,
    augment=True
)

train_loader, val_loader = data_config.get_loaders()

model = create_model('resnet18', num_classes=20, input_channels=1)

training_config = TrainingConfig(
    epochs=30,
    lr=1e-3,
    save_dir="./runs/my_experiment"
)

history = train_model(model, train_loader, val_loader, training_config, model_name="resnet18")
```

## ⚙️ Configuration Presets

See `configs.py` for pre-defined configurations:

- **QUICK_TEST_CONFIG**: 20 categories, 10 epochs (fast testing)
- **FULL_COMPARISON_CONFIG**: 50 categories, 30 epochs (all 3 models)
- **MLP_CONFIG**: 345 categories, 50 epochs (MLP only)
- **RESNET_CONFIG**: 345 categories, 50 epochs (ResNet-18 only)
- **VIT_CONFIG**: 345 categories, 50 epochs (ViT only)
- **HIGH_PERF_CONFIG**: 345 categories, 100 epochs, 50K samples/class

## 📊 Visualization

```python
from src.utils import plot_training_curves, compare_models, load_history

# Load histories
histories = {
    'mlp': load_history('./runs/mlp_history.json'),
    'resnet18': load_history('./runs/resnet18_history.json'),
    'vit': load_history('./runs/vit_history.json')
}

# Plot training curves
plot_training_curves(histories, save_path='./training_curves.png')

# Compare models
compare_models(results)
```

## 🔧 Customization

### Custom Categories
```python
from src.data import ALL_QUICKDRAW_CATEGORIES

# Select specific categories
my_categories = ['cat', 'dog', 'airplane', 'car', 'tree']

config = ExperimentConfig(
    categories=my_categories,
    num_classes=len(my_categories),
    epochs=20
)
```

### Custom Model Architecture
```python
# MLP with custom hidden layers
model = create_model(
    'mlp',
    num_classes=345,
    hidden_dims=(1024, 512, 256, 128),
    dropout=0.4
)

# ViT with custom architecture
model = create_model(
    'vit',
    num_classes=345,
    dim=384,
    depth=8,
    heads=12,
    mlp_dim=768,
    dropout=0.1
)
```

## 📝 Key Changes from Thesis Code

**Removed:**
- All autoencoder models (VanillaAE, JLAE, VAE, LoRAE)
- Johnson-Lindenstrauss (JL) projection layers
- Graph neural networks
- Text transformers
- Advanced evaluation metrics (LPIPS, OOD detection)
- Benchmarking utilities
- CelebA, FFHQ, Places365 datasets
- Triton/CUDA kernels

**Kept:**
- Basic ResNet-18 blocks (adapted for 28×28)
- Vision Transformer implementation
- Simple MLP classifier
- Training loop with checkpointing
- Basic visualization utilities
- Model comparison tools

## 💾 Output Structure

```
runs/
└── experiment_name/
    ├── mlp_best.pth              # Best checkpoint
    ├── mlp_final.pth             # Final checkpoint
    ├── mlp_history.json          # Training history
    ├── mlp_epoch10.pth           # Periodic checkpoint
    ├── resnet18_best.pth
    ├── resnet18_final.pth
    ├── resnet18_history.json
    ├── vit_best.pth
    ├── vit_final.pth
    └── vit_history.json
```

## 🎯 Typical Workflow

1. **Start with quick test** to verify everything works
2. **Run full comparison** on 50 categories to compare models
3. **Select best model** and train on full 345 categories
4. **Analyze results** with visualization tools
5. **Fine-tune** hyperparameters for final submission

## 📚 Additional Resources

- Quick, Draw! Dataset: https://github.com/googlecreativelab/quickdraw-dataset
- Original Data: https://quickdraw.withgoogle.com/data

## 🔍 Troubleshooting

**Issue**: Out of memory error
- **Solution**: Reduce batch size or use smaller models

**Issue**: ViT not working
- **Solution**: Install einops: `pip install einops`

**Issue**: Download fails
- **Solution**: Check internet connection and firewall settings

**Issue**: Slow data loading
- **Solution**: Reduce `max_per_class` or number of categories

## 📄 Backup Files

All original files have been backed up with `.backup` extension:
- `models.py.backup`
- `data.py.backup`
- `training.py.backup`
- `configs.py.backup`
- `utils.py.backup`

You can restore them if needed.
