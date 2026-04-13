# Two-Stage Global CP Refiner Implementation

## Overview

This branch implements a **two-stage architecture** for sinogram prediction:

1. **Stage 1 (Patch-wise)**: Existing `G1TransUnet` backbone predicts patch-by-patch, with full-sequence reconstruction via Hann overlap-add stitching. This stage remains **unchanged** in principle.

2. **Stage 2 (Global Refiner)**: New lightweight `GlobalCPRefiner1D` module refines the full stitched sinogram along the CP (rotation) axis only, learning to correct global coherence without reprocessing the DRR.

## Key Architecture Changes

### network.py

**New Components:**

- `_TemporalResBlock1D`: Lightweight 1D residual block with dilated convolutions along CP axis.
- `GlobalCPRefiner1D`: Global refiner module operating on [B, 1, Ncp, 64] tensors:
  - Input preprocessing: squeeze channel, transpose to [B, 64, Ncp]
  - Optional conditioning: angle/position embeddings + FiLM
  - 4 temporal residual blocks with dilations [1, 2, 4, 8]
  - Output projection initialized to zero (residual correction)
  - Hard gating: respects null-sample masking from stage 1
  - Alpha parameter: learned scaling of correction (init=0 for identity start)

- `TwoStageSinoModel`: Composite wrapper combining backbone + optional refiner:
  - `forward()`: delegates to `predict_patch()` (stage 1 only, for training compatibility)
  - `predict_patch()`: backbone forward for patch-wise training
  - `refine_full()`: applies global refiner with null masking
  - `get_refiner_parameters()`: exposes refiner params for selective optimization

### dataloader_patches.py

**New Loaders:**

- `get_full_sequence_loader(split)`: Creates a full-sequence DataLoader (no patching, no jitter) for train/val/test splits
  - Uses `SinogramValDataset` on the requested split
  - Returns batch_size=1 by default (full sequences)
  - Supports split_json manifest or ratio-based splitting

- `get_full_train_loader()`: Convenience alias for train split full-sequence loading

### model_simplified.py

**Configuration Extensions:**

- `ModelConfig`: Added refiner hyperparameters:
  - `use_global_refiner`: enable/disable the refiner
  - `refiner_hidden`, `refiner_layers`, `refiner_kernel_size`, `refiner_dilations`
  - `refiner_dropout`, `refiner_cond_dim`, `refiner_alpha_init`

- `TrainingConfig`: Added two-stage orchestration:
  - `training_mode`: `'single_stage'` (default) or `'two_stage'`
  - `stage1_epochs`: number of patch-wise backbone training epochs
  - `stage2_epochs`: number of full-sequence refiner training epochs
  - `stage2_learning_rate`: separate learning rate for stage 2
  - `freeze_backbone_stage2`: freeze backbone during stage 2 (recommended=true)
  - `lambda_refiner_delta`: L1 regularization weight on correction magnitude (default=1e-3)
  - `train_full_batch_size`: batch size for stage 2 (default=1)

**Model Building:**

- `_build_model()`: Now creates `TwoStageSinoModel` instead of bare `G1TransUnet`
- `_setup_training()`: Creates a separate `optimizer_stage2` if refiner parameters exist

**Training Flow:**

- `train(train_loader, val_loader, loss_fn, train_full_loader=None)`:
  - Automatically routes to stage-1 or stage-2 epoch based on epoch index
  - Requires `train_full_loader` if `training_mode='two_stage'`
  - Stage 1 runs for `stage1_epochs` on patches
  - Stage 2 runs for `stage2_epochs` on full sequences

- `_prepare_stage2()`: Freezes backbone (if requested), activates refiner optimizer
- `_train_epoch()`: Standard patch-wise training (unchanged)
- `_train_epoch_refiner()`: New stage-2 training loop:
  - Backbone in `eval()` + `torch.no_grad()`
  - Computes `y_base_full` from backbone (no grad)
  - Applies refiner → `y_refined`
  - Loss = `loss_main(y_refined, y_true) + lambda_delta * L1(y_refined, y_base)`
  - Logs both base and refined metrics for comparison

- `_validate_epoch()`: Now computes both base and refined predictions if refiner is enabled
  - Logs separate metrics for `*_base` versions

**Stitching & Gating:**

- `predict_full_base()`: Stage-1 stitching (was `predict_full`, now delegated)
  - Patch backbone forward via `self.model.predict_patch()`
  - Hann overlap-add accumulation
  - Hard gating: forces null samples to zero

- `predict_full(return_base=False)`: Full two-stage pipeline
  - Calls `predict_full_base()` → `y_base_full`
  - Optionally applies refiner → `y_final_full`
  - Returns `(y_final_full, denom_acc, y_base_full)` if `return_base=True`

- `_compute_null_batch_mask()`: Detects entirely null samples for hard gating

**Inference & Saving:**

- `inference()`: Now saves:
  - `y_pred.npy`: final output (refined if refiner enabled)
  - `y_pred_base.npy`: base stitched output (if refiner enabled)
  - `y_pred_refined.npy`: refined output (if refiner enabled)
  - Same metadata files as before (angles, positions, film, etc.)

**Checkpointing:**

- `save_checkpoint()`: Includes `optimizer_stage2_state_dict` if it exists
- `load_checkpoint()`: Loads with `strict=False` to support legacy checkpoints without refiner keys

### train.py

**Orchestration:**

- Detects `training_mode='two_stage'` from config
- If two-stage mode, builds `train_full_loader` via `get_full_sequence_loader(split='train',...)`
- Passes both `train_loader` and `train_full_loader` to `model.train()`

### configs/base.yaml

**New Keys Added:**

```yaml
model:
  use_global_refiner: true
  refiner_hidden: 128
  refiner_layers: 4
  refiner_kernel_size: 5
  refiner_dilations: [1, 2, 4, 8]
  refiner_dropout: 0.05
  refiner_cond_dim: 32
  refiner_alpha_init: 0.0

training:
  training_mode: two_stage
  stage1_epochs: 160
  stage2_epochs: 40
  stage2_learning_rate: 1.0e-4
  freeze_backbone_stage2: true
  lambda_refiner_delta: 1.0e-3
  train_full_batch_size: 1
```

Note: `use_transformer` remains **false** (this refiner replaces the idea of an internal bottleneck transformer).

## What Remains Unchanged

- **G1TransUnet backbone**: Completely intact, no modifications
- **Patch-wise training**: Still uses the original `SinogramPatchAugmentedDataset` with jitter and augmentation
- **Validation/inference on val/test**: Uses `SinogramValDataset` (full sequences, batch_size=1)
- **Hard gating rule**: Preserved, applied after refinement
- **Loss functions**: Reused as-is (no new loss class needed)
- **EMA and checkpointing**: Enhanced to support both stage optimizers

## Migration Path

1. **Old single-stage mode** (recommended for baseline):
   ```yaml
   training_mode: single_stage  # or omit, defaults to single_stage
   use_global_refiner: false
   ```
   → Behaves exactly like before, ignoring refiner code entirely.

2. **New two-stage mode**:
   ```yaml
   training_mode: two_stage
   use_global_refiner: true
   stage1_epochs: 160
   stage2_epochs: 40
   ```
   → First 160 epochs train patch backbone, next 40 refine globally.

3. **Legacy checkpoints**: Auto-loaded with `strict=False`, missing refiner keys are silently ignored.

## Testing & Validation

- Smoke test passed: Model initializes with `use_global_refiner=True`
- All imports validated: `TwoStageSinoModel`, `GlobalCPRefiner1D`, `get_full_sequence_loader`
- Syntax check passed on all modified files
- Configuration parsing validated

## Performance Considerations

- **Stage 1 cost**: Unchanged (patch-wise, high cost per sample)
- **Stage 2 cost**: Very low (only operates on [B, 64, Ncp], ~1% of U-Net params)
- **Memory**: Slightly increased (maintains both base and refined predictions during validation)
- **Inference time**: Stage 2 adds ~1-2% overhead (small 1D network)

## Next Steps (Optional Enhancements)

1. Tune `refiner_hidden`, `refiner_layers`, `refiner_dilations` based on validation gains
2. Experiment with `freeze_backbone_stage2=false` for fine-tuning (higher cost, potentially better results)
3. Monitor `loss_delta` and refiner alpha to verify it's not overlearning
4. Compare `y_pred_base` vs `y_pred_refined` metrics to quantify the refiner contribution
5. Consider LoRA or other parameter-efficient fine-tuning for stage 2 if needed

---

**Branch**: `feature/two-stage-global-refiner`  
**Status**: Ready for testing and integration

