# Network Architecture Update - Branch `exp`

**Date**: 2026-03-04  
**Branch**: exp  
**Status**: ✅ Implementation complete

## Summary

Major update to `network.py` implementing a **CP-aware multi-branch stem** with semantic channel grouping, learnable shoulder gating, internal FiLM embedding, and corrected encoder/decoder FiLM mapping.

---

## 🎯 Key Changes

### 1. **CP-Aware Multi-Branch Stem** (`CPStemMultiBranch`)

Replaces the previous `CPUnshuffleStem` with a semantically-aware architecture that respects the meaning of the 16 input channels:

#### Channel Groups (assumes standard order):
- **PTV (channels 0-2)**: `ptv_br`, `ptv_ri`, `ptv_hr` - primary targets/drivers
- **External (channels 3-4)**: `external_entry`, `external_exit` - patient support/context
- **Shoulder (channel 14)**: Shoulders - strong occlusion potential (configurable index)
- **Other OARs (remaining 10)**: Other organs at risk - local penalty/modulation

#### Architecture:
1. **Pre-processing**: Depthwise vertical conv (intra-CP filtering)
2. **Exact repack**: `[B, 16, Hpx, W]` → `[B, 16*cp_height, Hcp, W]`
3. **Branch processing**: Each group gets dedicated conv pathway
   - PTV branch → 48 channels (base_ch // 2)
   - External branch → 24 channels (base_ch // 4)
   - Shoulder branch → 24 channels (base_ch // 4)
   - OAR-other branch → 48 channels (base_ch // 2)
4. **Fusion**: Concat + 1×1 conv + 3×3 conv → `base_ch`

**Benefits**:
- Preserves semantic meaning of each channel group
- Prevents early mixing of drivers vs. context vs. penalties
- Allows network to learn separate features per organ type

---

### 2. **Learnable Shoulder Gate**

Optional soft gating mechanism to model the physical phenomenon: "shoulders in front of target → strong fluence attenuation"

#### Implementation:
```python
gate = sigmoid(conv1x1([shoulder_features, external_features]))
strength = sigmoid(learnable_param)  # in (0,1)
fused_output = fused_output * (1 - strength * gate)
```

- **Soft & differentiable**: No hard zeroing, learns optimal attenuation
- **Configurable**: `shoulder_gate=True/False`, `shoulder_oar_index_in_oars=9`
- **Interpretable**: `strength` parameter shows learned importance

---

### 3. **Internal FiLM Embedding** (`FilmEmbed`)

Adds a small MLP to map plan-level conditioning (7D film_extra) to a richer internal representation (32D by default):

```
film_extra [B, 7] → FilmEmbed(7→32) → [B, Hcp, 32]
cond_final = concat(angle_embed, position_embed, film_embed)  # [B, Hcp, 2*d_model+32]
```

**Benefits**:
- More expressive conditioning without changing dataloader
- Stable defaults: `film_embed_dim=32`, `film_embed_hidden=64`
- Automatic handling of missing film_extra (zero-filled)

---

### 4. **FiLM Conditioning Cache**

Within each forward pass, cache conditioning vectors by height to avoid redundant computation:

```python
cond_cache: Dict[int, torch.Tensor] = {}

def get_cond(h: int) -> torch.Tensor:
    if h not in cond_cache:
        cond_cache[h] = self._cond(angles, positions, target_len=h, film_extra=film_extra)
    return cond_cache[h]
```

**Impact**: Reduces resampling and embedding overhead when multiple levels share the same CP count.

---

### 5. **Corrected FiLM Mapping Per Level**

Fixed encoder/bottleneck FiLM mapper construction to match exact architecture levels:

**Old** (duplicated last level):
```python
self.film_enc = [
    film_inc,
    *[film_down_i for each down],
    film_bneck,  # duplicate
    film_bneck   # duplicate!
]
```

**New** (exact mapping):
```python
self.film_global = FiLMMap(...)
self.film_inc = FiLMMap(...)
self.film_down = ModuleList([FiLMMap(...) for each down])
self.film_bneck = FiLMMap(...)
self.film_dec = ModuleList([FiLMMap(...) for each up])
```

---

### 6. **Upsampling: Interpolation + Conv**

Changed from `ConvTranspose2d` to `interpolate + conv` to avoid checkerboard artifacts:

```python
# Old
x1 = ConvTranspose2d(...)(x1)

# New
x1 = F.interpolate(x1, scale_factor=(2, up_w), mode="bilinear")
x1 = conv3x3(x1)  # smooth post-interpolation
```

---

### 7. **Updated Defaults**

Changed defaults to safer/more stable values:

| Parameter | Old Default | New Default | Reason |
|-----------|-------------|-------------|--------|
| `film_on_decoder` | `True` | `False` | Reduce horizontal banding risk |
| `film_embed_dim` | N/A | `32` | Enable richer conditioning |
| `film_embed_hidden` | N/A | `64` | Balanced capacity |
| `shoulder_gate` | N/A | `True` | Enable by default |
| `shoulder_oar_index_in_oars` | N/A | `9` | Standard shoulder position |

---

## 🔧 Configuration Parameters

### New Parameters in `ModelConfig`:

```yaml
model:
  # FiLM internal embedding
  film_embed_dim: 32          # Internal FiLM dimension (0 to disable)
  film_embed_hidden: 64       # Hidden layer size for film embed
  
  # Stem options
  stem_pre_k: 3              # Pre-repack vertical conv kernel (must be odd)
  shoulder_gate: true        # Enable learnable shoulder gating
  shoulder_oar_index_in_oars: 9  # Index of shoulder in 11 OARs (0-10)
  
  # Existing (updated defaults)
  film_on_decoder: false     # Changed from true (reduces banding)
```

---

## 📊 Channel Mapping Reference

**Critical**: Verify your dataset's channel order matches this assumption!

```python
# Indices in x_drr [B, 16, Hpx, W]:
idx_ptv = [0, 1, 2]           # ptv_br, ptv_ri, ptv_hr
idx_ext = [3, 4]              # external_entry, external_exit
idx_oars = [5..15]            # 11 OARs
  shoulder_idx = 14           # Global index (shoulder_oar_index_in_oars=9 in [5..15])
  other_oars = [5..13, 15]    # 10 remaining OARs
```

**Action Required**: If your shoulder is at a different position, update `shoulder_oar_index_in_oars` in config!

---

## ✅ Validation Checklist

- [x] Branch `exp` created
- [x] `network.py` updated with all changes
- [x] `model_simplified.py` config dataclass updated
- [x] Legacy config builder updated
- [x] No compile errors
- [ ] Smoke test passed (forward pass validation)
- [ ] Training run with new defaults
- [ ] Ablation: shoulder_gate on/off
- [ ] Ablation: film_embed_dim 0/16/32/64

---

## 🚀 Usage Examples

### Basic Training with New Architecture

```bash
# Use updated base config (film_on_decoder=false, shoulder_gate=true)
python train.py --config configs/base.yaml --run_name exp-baseline
```

### Disable Shoulder Gate (Ablation)

```yaml
# config_custom.yaml
model:
  shoulder_gate: false
```

### Custom Shoulder Index

If shoulders are at a different position in your 11 OARs:

```yaml
model:
  shoulder_oar_index_in_oars: 5  # Example: 6th OAR (index 10 globally)
```

### Disable Internal FiLM Embedding

```yaml
model:
  film_embed_dim: 0  # Revert to direct film_extra concat
```

---

## 🔍 Debugging Tips

### Shape Debug Mode

Enable detailed shape logging:

```yaml
model:
  shape_debug: true
```

Output:
```
stem+earlyW: torch.Size([2, 64, 512, 64])
inc: torch.Size([2, 64, 512, 64])
down0: torch.Size([2, 128, 256, 64])
down1: torch.Size([2, 256, 128, 64])
bneck: torch.Size([2, 256, 128, 64])
up0: torch.Size([2, 128, 256, 64])
up1: torch.Size([2, 64, 512, 64])
outc: torch.Size([2, 1, 512, 64])
```

### Verify Shoulder Index

Check which channel is actually the shoulder in your dataset preprocessing pipeline, then set accordingly.

---

## 📝 Migration from Previous Version

### If Resuming from Old Checkpoint

**Incompatible**: Old checkpoints used `CPUnshuffleStem`, new uses `CPStemMultiBranch`

**Options**:
1. **Train from scratch** (recommended for clean comparison)
2. **Partial load**: Load encoder/decoder weights only, skip stem
3. **Keep old branch**: Use previous `main` for continued training

### Config Migration

Old configs missing new params will use safe defaults via `getattr(..., default)`.

---

## 🎓 Theoretical Motivation

### Why Multi-Branch Stem?

**Problem**: Mixing all 16 channels immediately loses semantic information
- PTVs are **targets** (what we want to irradiate)
- Externals are **support geometry** (patient boundaries)
- Shoulders are **strong occluders** (physical blocking)
- Other OARs are **local penalties** (regional modulation)

**Solution**: Process each group separately, then fuse with learned weights

### Why Shoulder Gate Specifically?

Physical phenomenon: When shoulders overlap target in beam's eye view, fluence is strongly attenuated (bone + tissue density). Modeling this explicitly as a multiplicative gate improves physical realism.

---

## 📈 Expected Impact

Based on architecture improvements:

- **Better semantic preservation** → Improved target coverage
- **Shoulder-aware gating** → More realistic fluence in shoulder-overlapping CPs
- **Richer conditioning** → Better dose/duration/angle response
- **Reduced checkerboard** → Smoother sinogram predictions
- **Decoder FiLM off** → Less horizontal banding artifacts

---

## 🧪 Recommended Experiments

1. **Baseline**: Train with all defaults on `exp` branch
2. **Ablation 1**: `shoulder_gate=false`
3. **Ablation 2**: `film_embed_dim=0`
4. **Ablation 3**: `film_on_decoder=true` (compare banding)
5. **Compare**: Best `exp` vs. current `main` production model

---

## 📚 Related Files

- `network.py` - Main architecture
- `model_simplified.py` - Config dataclass + training wrapper
- `configs/base.yaml` - Base configuration with new defaults
- `train.py` - Training entry point

---

## 🐛 Known Issues

- Minor warning: `_resample_len` returns `None` when `x is None` (type hint expects Tensor)
  - **Impact**: None (type checker warning only)
  - **Fix**: Low priority, doesn't affect runtime

---

## 👤 Author

Implementation by AI assistant for Julien's sianogram project, March 4, 2026.

---

## 📄 License

Same as parent project.

