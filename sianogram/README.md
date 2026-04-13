# Sianogram: U-Net 2D Sinogram Prediction with FiLM Conditioning

Deep learning model for predicting radiation therapy sinograms with CP-aware multi-branch architecture and learnable occlusion modeling.

---

## 🎯 Quick Start

### Installation
```bash
conda env create -f environment.yml
conda activate Sinogramme
```

### Quick Test (1 epoch - ~5-10 min, MSELoss)
```bash
python train.py \
  --config configs/base.yaml \
  --override configs/quicktest.yaml \
  --run_name quicktest
```
*Note: `base.yaml` uses MSELoss by default. Use `quicktest.yaml` override for 1 epoch only.*

### Production Run (200 epochs, BalancedLogSpectralLoss)
```bash
python train.py \
  --config configs/run_balancedlog.yaml \
  --run_name prod-baseline
```

### Inference (best/latest checkpoint)
```bash
# Infer from a previous run (best checkpoint)
python inference.py \
  --resume-from runs/20260304_140849__newmodel* \
  --checkpoint-type best \
  --out-dir inference_outputs/prod_best

# Same with latest checkpoint
python inference.py \
  --resume-from runs/20260304_140849__newmodel* \
  --checkpoint-type latest \
  --out-dir inference_outputs/prod_latest
```

### Inference + RTPLAN injection (optional)
```bash
python inference.py \
  --resume-from runs/20260304_140849__newmodel* \
  --inject-rtplan \
  --dicom-root /mnt/LeGrosDisque/Julien/sianogramme/dicom_data \
  --out-dir inference_outputs/with_rtplan
```

### Exp Branch (new architecture)
```bash
git checkout exp
python train.py \
  --config configs/base.yaml \
  --override configs/run_exp_baseline.yaml \
  --run_name exp_baseline
```

---

## 📚 Documentation

### Quick Start & Launch
- **🚀 [LAUNCHING_GUIDE.md](LAUNCHING_GUIDE.md)** - How to launch tests, modify parameters, and run experiments
- **📖 [configs/README.md](configs/README.md)** - Configuration and loss functions reference

### Architecture & Development
- **🏗️ [documentation/NETWORK_EXP_CHANGES.md](documentation/NETWORK_EXP_CHANGES.md)** - Exp branch: CP-aware stem + shoulder gating

### Additional Documentation
- **📂 [documentation/](documentation/)** - Technical analysis and detailed guides

---

## 🧪 Loss Functions

```bash
# 1. BalancedLogSpectralLoss (recommended)
python train.py --config configs/run_balancedlog.yaml --run_name balanced

# 2. SparseFocalSpectralLoss
python train.py --config configs/run_sparsefocal.yaml --run_name focal

# 3. SparseSinoLoss
python train.py --config configs/run_sparsesino.yaml --run_name sino

# 4. GigaUltimateLoss
python train.py --config configs/run_gigaultimate.yaml --run_name giga
```

---

## 🏗️ Architecture

### Main Branch (Production)
- Standard U-Net 2D with CP-aware input repack
- FiLM conditioning (angle + position + plan-level features)
- Configurable depth, channels, bottleneck

### Exp Branch (Development)
Improvements:
- **CP-Aware Multi-Branch Stem**: Semantic channel grouping
- **Learnable Shoulder Gate**: Occlusion modeling
- **Internal FiLM Embedding**: Richer conditioning (7D → 32D)
- **Anti-Checkerboard Upsampling**: Interpolation + conv

---

## 📁 Project Structure

```
# Core pipeline
├── train.py                      # Training entrypoint
├── inference.py                  # Inference entrypoint (predictions + RTPLAN)
├── model_simplified.py           # Model + training/inference wrapper
├── network.py                    # U-Net architecture
├── dataloader_patches.py         # Data loading
├── losses.py                     # Loss functions
├── sino_metrics.py               # Sinogram-specific metrics
├── run_utils.py                  # Run directory helpers
├── util.py                       # Misc utilities
├── rtplan_injector.py            # DICOM RTPLAN injection
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment

# Configuration
├── configs/                      # YAML configurations

# Documentation
├── README.md                     # This file
├── LAUNCHING_GUIDE.md            # How to launch runs
├── documentation/                # Architecture docs, analysis notes

# Standalone tools (not part of core pipeline)
├── tools/
│   ├── analyze_training_log.py   # Parse training.log files
│   ├── tensorboard_analyzer.py   # Extract TensorBoard metrics
│   ├── visualize_threshold_histogram.py  # Sinogram threshold viz
│   └── launch_exp_run.sh         # Launch script for exp branch

# Tests
├── tests/
│   └── test_train_smoke.py       # Smoke tests for train.py

# Other
├── preprocessing/                # Data preprocessing pipeline
├── legacy/                       # Archived old code
└── runs/                         # Training run outputs (gitignored)
```

---

## 📊 Training Output

```
runs/TIMESTAMP_runname/
├── config.yaml
├── training.log
├── checkpoints/
│   ├── best.pth
│   └── latest.pth
├── TensorBoard/
└── train_visuals/               # epoch_X_it_Y_patient_Z.png
    val_visuals/
```

### Monitor
```bash
tensorboard --logdir runs/
tail -f runs/*/training.log
```

---

## 📋 Branches

| Branch | Status | Purpose |
|--------|--------|---------|
| `main` | ✅ Production | Stable pipeline (train + inference + RTPLAN) |
| `exp` | 🔬 Development | New architecture (CP-aware stem, shoulder gate) |
| `rework_test` | ✅ Merged → main | Inference pipeline + RTPLAN injection |
| `rework_nettoyage` | ✅ Merged → main | Root cleanup (tools/, documentation/) |

---

**Last Updated**: March 19, 2026

