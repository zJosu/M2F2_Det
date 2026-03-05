# AIGI Detection — AI-Generated Image Detection

Reproducible implementations of three complementary AI-generated image detection methods:

| Method | Approach | Training Required |
|--------|----------|-------------------|
| **CIFAKE** | Lightweight CNN + Grad-CAM | ✅ Supervised |
| **AIDE** | Multi-expert (DCT patches + CLIP semantics) + gated fusion | ✅ Supervised |
| **ZED** | Zero-shot entropy-based (PixelCNN codec) | ❌ Zero-shot |

## Quick Start

```bash
# Install dependencies
cd aigi_detection
pip install -e ".[dev]"

# Download CIFAKE dataset (~500MB)
python -m datasets.download_cifake --output data/CIFAKE

# Run full pipeline (smoke test — ~5 min)
python -m experiments.run_all --config configs/default.yaml --smoke-test

# Run full training + evaluation
python -m experiments.run_all --config configs/default.yaml

# Run only evaluation (with existing checkpoints)
python -m experiments.run_all --config configs/default.yaml --eval-only
```

## Project Structure

```
aigi_detection/
├── configs/default.yaml          # Central configuration
├── datasets/
│   ├── cifake_dataset.py         # CIFAKE PyTorch Dataset
│   └── download_cifake.py        # Dataset downloader
├── models/
│   ├── cifake/
│   │   ├── cnn.py                # CIFAKENet (8-layer CNN)
│   │   └── gradcam.py            # Grad-CAM heatmap generator
│   ├── aide/
│   │   ├── patch_expert.py       # DCT-based patch artifact expert
│   │   ├── clip_expert.py        # CLIP semantic feature expert
│   │   ├── fusion.py             # Gated expert fusion module
│   │   └── aide_model.py         # Composed AIDE model
│   └── zed/
│       ├── codec.py              # PixelCNN autoregressive codec
│       ├── detector.py           # ZED zero-shot detector
│       └── anomaly_maps.py       # Anomaly map visualisation
├── training/
│   ├── train_cifake.py           # CIFAKE training loop
│   └── train_aide.py             # AIDE training loop
├── evaluation/
│   ├── evaluate.py               # Unified evaluation
│   ├── robustness.py             # Perturbation robustness testing
│   └── compare.py                # Comparison plots
├── explainability/
│   ├── maps.py                   # Unified explanation map generator
│   ├── consistency.py            # Map consistency & stability metrics
│   └── visualize.py              # Comparison visualisations
├── experiments/
│   └── run_all.py                # Master experiment runner
├── utils/
│   ├── seed.py                   # Reproducibility
│   ├── logging.py                # TensorBoard + CSV logging
│   ├── metrics.py                # Accuracy, F1, AUC
│   └── transforms.py             # Data transforms & perturbations
├── tests/
│   ├── test_models.py            # Model architecture tests
│   ├── test_metrics.py           # Metric computation tests
│   └── test_transforms.py        # Transform tests
└── pyproject.toml                # Dependencies
```

## Methods

### CIFAKE (CNN Baseline)
- **Architecture**: 4 conv blocks (32→256 channels) + 2 extra conv layers + classifier
- **Input**: 32×32 RGB images
- **Explainability**: Grad-CAM heatmaps showing classifier attention regions
- **Training**: BCE loss, Adam, cosine LR, ~30 epochs

### AIDE (Multi-Expert)
- **Patch Expert**: Extracts DCT frequency features from image patches for artifact detection
- **CLIP Expert**: Frozen ViT-B/32 + learnable projection for semantic analysis
- **Fusion**: Learned gating network with per-expert softmax weights
- **Input**: 224×224 (upscaled for CLIP compatibility)

### ZED (Zero-Shot)
- **Codec**: PixelCNN with masked convolutions modelling P(pixel | context)
- **Detection**: Anomaly = |NLL − Entropy| → regions where coding cost deviates from expected
- **No fake training data**: Threshold calibrated from real images only

## Evaluation

All methods are compared on:
- **Accuracy, F1-score, ROC-AUC**
- **Robustness**: JPEG (Q=50/70/90), Gaussian noise (σ=0.01/0.05/0.1), cropping (80%/60%)
- **Explainability**: Map consistency, stability under augmentation, cross-method IoU overlap

## Configuration

All settings are in `configs/default.yaml`. Key parameters:
- `seed`: Random seed for reproducibility (default: 42)
- `device`: "auto", "cuda", or "cpu"
- Model-specific: learning rate, batch size, epochs, architecture params

## Tests

```bash
python -m pytest tests/ -v
```
