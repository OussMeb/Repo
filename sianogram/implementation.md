# Implementation Notes - Halo-First Direction

## 1) Scope and intent

This document captures the current implementation direction requested for this branch:

1. Prioritize **halo patching** for patch-wise training and stitching.
2. Keep the optional global two-stage refiner code in the repository, but **disable it for this experiment series**.
3. Add an optional lightweight **local patch CP-only head** (post patch output), disabled by default.

The goal is to improve intrinsic patch quality first (especially border behavior) before reintroducing any global refinement.

---

## 2) High-level design

### 2.1 Halo patching

We move from classic patching to:

- `patch_in_cp > patch_out_cp`
- `patch_in_cp = patch_out_cp + 2 * halo_cp`

Recommended baseline values used in config:

- `patch_out_cp = 512`
- `halo_cp = 128`
- `patch_in_cp = 768`

Training supervision is applied only on the center region (core), while halo borders are context only.

### 2.2 Local CP-only head (optional)

A lightweight residual CP-only head can be applied directly on patch predictions:

- Input patch prediction: `[B, 1, Ncp, 64]`
- Internal transform: `[B, 64, Ncp]`
- 1D temporal residual processing on CP axis
- Residual output: `y = y0 + alpha * delta`

This is local to each patch and does not depend on full patient sequence stitching.

### 2.3 Global refiner status

- Global refiner modules remain in codebase.
- For this experiment track, defaults are:
  - `training_mode: single_stage`
  - `use_global_refiner: false`

---

## 3) Data and tensor contracts

### 3.1 Stage-1 patch training (halo)

Input batch tensors:

- `x_drr`: `[B, Cin, patch_in_cp * cp_height, W_in]`
- `angles`: `[B, patch_in_cp, 1]`
- `positions`: `[B, patch_in_cp, 1]`
- `y_sino`: `[B, 1, patch_in_cp, 64]`
- `core_start_cp`, `core_end_cp` (metadata per batch)

Core supervision region:

- `y_pred_core = y_pred_full[:, :, core_start_cp:core_end_cp, :]`
- `y_true_core = y_true_full[:, :, core_start_cp:core_end_cp, :]`

Loss and train metrics are computed on core only.

### 3.2 Validation/inference full sequence

Model still reconstructs full sequence through sliding windows, but now in halo mode:

- Extract input windows of length `patch_in_cp`
- Predict full patch
- Keep only center `patch_out_cp` region from each patch prediction
- Stitch only center regions to final sequence

Halo borders are never injected directly into final reconstructed output.

---

## 4) File-by-file implementation

## 4.1 `dataloader_patches.py`

### Added/updated batch metadata

- `BATCH_KEYS` now includes:
  - `core_start_cp`
  - `core_end_cp`

- `_build_batch_dict(...)` now accepts optional core bounds and stores them as tensors.

### Halo-aware train dataset

`SinogramPatchAugmentedDataset` now supports:

- `patch_in_cp`
- `patch_out_cp`
- `halo_cp`

Validation checks enforce:

- `patch_in_cp >= patch_out_cp`
- `patch_in_cp == patch_out_cp + 2 * halo_cp`

Sampling behavior:

- Sampling stride/count is based on `patch_out_cp` (useful region).
- Input extraction is based on `patch_in_cp` (context region).
- Left/right out-of-range halo context is padded (X with zeros, geometry with edge behavior via helper).
- Returned core indices are fixed in local patch frame:
  - `core_start_cp = halo_cp`
  - `core_end_cp = halo_cp + patch_out_cp`

### Loader factories

`get_patch_loaders(...)` now accepts and forwards:

- `patch_in_cp`
- `patch_out_cp`
- `halo_cp`

Backward compatibility:

- Existing `patch_cp` remains accepted as a legacy alias.
- If halo params are not provided, defaults are derived so old behavior still works.

`get_full_sequence_loader(...)` and `get_test_loader(...)` also accept halo params and align minimum CP padding with `patch_in_cp` for compatibility with halo stitching path.

---

## 4.2 `model_simplified.py`

### Config updates

`DataConfig` now includes:

- `patch_in_cp: Optional[int]`
- `patch_out_cp: Optional[int]`
- `halo_cp: int`
- `patch_cp` kept as legacy alias

`__post_init__` resolves coherent values and enforces:

- `patch_in_cp == patch_out_cp + 2 * halo_cp`

`ModelConfig` now includes local patch head options:

- `use_patch_cp_head`
- `patch_cp_head_hidden`
- `patch_cp_head_layers`
- `patch_cp_head_kernel_size`
- `patch_cp_head_dilations`
- `patch_cp_head_dropout`
- `patch_cp_head_alpha_init`

Legacy config factory mapping updated accordingly.

### Stage-1 train loop core-only supervision

In `_train_epoch(...)`:

- Reads `core_start_cp/core_end_cp` from batch.
- Crops predictions and targets to core before loss.
- Crops `x_drr` in CP-pixel space for losses needing `x_drr`.
- Computes train metrics on core region only.

### Halo-aware full reconstruction

In `predict_full_base(...)`:

- Uses `patch_in_cp`, `patch_out_cp`, `halo_cp`.
- Builds sliding windows over output core positions.
- Extracts wider input context windows.
- Runs patch prediction on full input window.
- Keeps only center output region for accumulation.
- Applies overlap blending on kept core region.
- Preserves hard gating behavior for null samples.

`predict_full(...)` remains wrapper logic (base + optional global refiner), but global refiner is off in current baseline config.

---

## 4.3 `network.py`

### New local patch CP head

Added `PatchCPHead1D`:

- Lightweight residual 1D CP-only head.
- Operates on patch outputs `[B,1,Ncp,64]`.
- Internal layout `[B,64,Ncp]`.
- Stack:
  - `Conv1d(64 -> hidden, k=1)`
  - residual temporal blocks (dilated 1D)
  - `Conv1d(hidden -> 64, k=1)`
- Residual output with learned scalar `alpha`, initialized to zero-like behavior.

### `G1TransUnet` wiring

- New optional init flags for local patch head.
- `from_configs(...)` reads these flags.
- If enabled, applies `self.patch_cp_head(...)` after `self.outc(...)`.

### Input height source

`G1TransUnet.from_configs(...)` now uses `data_cfg.patch_in_cp` (fallback to `patch_cp`) for input CP height sizing.

---

## 4.4 `train.py`

- Consistency check now uses `patch_in_cp * cp_height_px`.
- Enforces halo parameter coherence.
- Passes `patch_in_cp/patch_out_cp/halo_cp` into loaders.

Two-stage support remains in codebase, but default experiment path now uses single-stage.

---

## 4.5 `inference.py`

- Test loader creation now forwards:
  - `patch_in_cp`
  - `patch_out_cp`
  - `halo_cp`

This ensures inference reconstruction is aligned with halo-aware patch extraction/reconstruction behavior.

---

## 4.6 `configs/base.yaml`

Baseline now set for halo-first experiments:

Data:

- `patch_cp: 768` (legacy alias)
- `patch_in_cp: 768`
- `patch_out_cp: 512`
- `halo_cp: 128`

Model:

- `use_global_refiner: false`
- local patch CP head params present
- `use_patch_cp_head: false` by default

Training:

- `training_mode: single_stage`

---

## 5) Experiments requested

## 5.1 Experiment A - Halo only

- Halo patching enabled
- Local patch head disabled
- Global refiner disabled

```bash
python train.py --config configs/base.yaml --run_name expA_halo_only
```

## 5.2 Experiment B - Halo + local patch CP head

Create a small override:

```bash
cat > /tmp/expB_patch_head.yaml <<'YAML'
model:
  use_patch_cp_head: true
YAML
```

Run:

```bash
python train.py --config configs/base.yaml --override /tmp/expB_patch_head.yaml --run_name expB_halo_plus_patchhead
```

## 5.3 Experiment C (later)

Not part of current priority.
Potentially re-enable global refiner after halo-first outcomes are validated.

---

## 6) Verification performed

The following checks were executed after the modifications:

1. Python compile checks on touched runtime files:

```bash
python -m py_compile /home/julien/PycharmProjects/sianogram/dataloader_patches.py
python -m py_compile /home/julien/PycharmProjects/sianogram/model_simplified.py
python -m py_compile /home/julien/PycharmProjects/sianogram/network.py
python -m py_compile /home/julien/PycharmProjects/sianogram/train.py
python -m py_compile /home/julien/PycharmProjects/sianogram/inference.py
```

2. Runtime smoke initialization with halo config and local patch head enabled (`cpu`) succeeded.

---

## 7) Notes and caveats

- Full-sequence reconstruction still uses overlap accumulation, but now only on kept center regions.
- For very short sequences, padding behavior remains compatible with existing val/test dataset strategy.
- Global refiner and two-stage machinery are still available in code; they are intentionally disabled in baseline config to isolate halo effects.

---

## 8) Quick toggle matrix

- Halo only baseline:
  - `training.training_mode: single_stage`
  - `model.use_global_refiner: false`
  - `model.use_patch_cp_head: false`

- Halo + local patch head:
  - same as above, except `model.use_patch_cp_head: true`

- Two-stage global refiner (not current priority):
  - `training.training_mode: two_stage`
  - `model.use_global_refiner: true`

---

## 9) Suggested next validation outputs

During comparison A vs B, focus on:

- Train/val loss stability
- CP continuity-related metrics
- Border/seam behavior in reconstructed full sequences
- Any leakage/background regressions

Keep A and B strictly isolated (same seed/split/epochs/loss) except the local patch head toggle.

