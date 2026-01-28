# Quick, Draw! Model Inference

This directory contains scripts for running inference with trained models.

## Usage

### Interactive Inference

Run the interactive inference script:
```bash
python guesser/inference.py
```

The script will:
1. Load all three trained models (MLP, ResNet-18, ViT)
2. Load the validation dataset
3. Prompt you to enter an image index to test
4. Show predictions from all three models
5. Display a visualization with the image and predictions
6. Save the visualization to `guesser/prediction_{idx}.png`

### Example Session

```
Using device: cuda

Loading Quick, Draw! dataset...
Dataset loaded: 50 categories

============================================================
Loading Models
============================================================
Loaded mlp model from models/mlp_best.pth
  - Epoch: 25
  - Val Accuracy: 78.50%
Loaded resnet18 model from models/resnet18_best.pth
  - Epoch: 30
  - Val Accuracy: 85.20%
Loaded vit model from models/vit_best.pth
  - Epoch: 35
  - Val Accuracy: 88.40%

============================================================
Running Inference
============================================================

Dataset has 5000 validation images
Enter image index (0-4999) or 'q' to quit: 42

True label: cat
MLP          → cat                   (confidence:  67.3%) ✓
ResNet-18    → cat                   (confidence:  89.5%) ✓
ViT          → cat                   (confidence:  94.2%) ✓

Visualization saved to: guesser/prediction_42.png
```

## Requirements

The script requires:
- Trained model checkpoints in `models/` directory:
  - `mlp_best.pth`
  - `resnet18_best.pth`
  - `vit_best.pth`
- Quick, Draw! dataset (automatically downloaded if not present)
- PyTorch, matplotlib, numpy

## Output

The script generates visualizations showing:
- The input image
- Predictions from each model
- Confidence scores
- Whether each prediction is correct (✓) or incorrect (✗)

Visualizations are saved to `guesser/prediction_{idx}.png`
