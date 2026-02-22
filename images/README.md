# Images Directory

This directory contains output images generated from model inference and policy comparison.

## Contents

### Policy Comparison Images
- `policy_comparison_*.png` - Visualizations showing all 9 trained policies (3 models × 3 policy types) and when each decided to guess during progressive stroke reveal

### Prediction Images
- `prediction_*.png` - Individual prediction visualizations from model inference

### Strokes Directory
- `strokes/` - Contains example stroke detection visualizations organized by category

## Generating New Images

To generate policy comparison images:
```bash
python src/run_trained_policies.py
```

The script will interactively let you test images and save comparison visualizations showing:
- Original image
- Partial images when each policy decided to guess
- Prediction accuracy for each policy
