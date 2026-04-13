# Configs Recommandées - Comparaison de Loss Functions

Ce dossier contient les configurations optimisées pour comparer 4 fonctions de loss sur le même modèle (base_ch=96).

## Configs Standalone (Recommandé)

Ces configs sont **complètes** et **identiques** sauf pour la loss. Elles utilisent toutes:
- `base_ch: 96` (boost capacité du modèle)
- `batch_size: 12` avec `accum_steps: 2` (batch effectif = 24)
- `patch_cp: 1024` (patches plus grands)
- `learning_rate: 8e-5`
- `scheduler_patience: 5` (plus patient)
- Même `random_seed: 44` et même `split_json` pour comparabilité stricte

### Fichiers

1. **`run_balancedlog.yaml`** - BalancedLogSpectralLoss (RECOMMANDÉ EN PREMIER)
   - Loss log-domain + gradients + spectre
   - Équilibre fond/pics sans ignorer les valeurs faibles
   - Paramètres: `k_log=20`, `w_min=0.20`, `p_w=0.35`

2. **`run_sparsefocal.yaml`** - SparseFocalSpectralLoss
   - Pour écraser les faux positifs de fond
   - Masque focal + weighted MSE + gradients + FFT
   - Paramètres: `w_fp=4.0`, `focal_gamma=2.0`, `mask_k=40.0`

3. **`run_sparsesino.yaml`** - SparseSinoLoss
   - Simple, robuste, efficace
   - Zones actives + pénalité FP avec rampe rapide
   - Paramètres: `w_neg=8.0`, `ramp_epochs=10`

4. **`run_gigaultimate.yaml`** - GigaUltimateLoss
   - "Couteau suisse" avec termes physiques optionnels
   - Weighted L1 + gradients + fluence + FFT 2D
   - Paramètres: `w_fluence_cp=0.05`, `w_ffl2d=0.03`

## Lancement

### Une config à la fois
```bash
python train.py --config configs/run_balancedlog.yaml --run_name balancedlog-test
python train.py --config configs/run_sparsefocal.yaml --run_name sparsefocal-test
python train.py --config configs/run_sparsesino.yaml --run_name sparsesino-test
python train.py --config configs/run_gigaultimate.yaml --run_name gigaultimate-test
```

### Les 4 en séquentiel
```bash
scripts/run_4_losses.sh
```

### Les 4 en parallèle (4 GPUs)
```bash
MODE=parallel GPUS=0,1,2,3 scripts/run_4_losses.sh
```

## Protocole de Comparaison (Important!)

Pour comparer proprement les 4 losses:

1. **Ne changez RIEN** entre les runs sauf le fichier config
2. **Même split**: utilisez le même `split_json`
3. **Même seed**: `random_seed: 44` dans toutes les configs
4. **Comparez à epochs fixes**: par exemple epoch 10, 20, 40, 80, 160
5. **Surveillez TensorBoard**: `tensorboard --logdir runs/`
6. **Si encore OOM**: baissez SEULEMENT `batch_size` à 4 (puis `accum_steps: 6`) ou 3 (puis `accum_steps: 8`)

## Note OOM Important

Les configs par défaut utilisent **`batch_size: 6`** + **`accum_steps: 4`** (batch effectif = 24).

Ceci a été testé OOM-safe sur GPU 24GB avec `base_ch: 96` + `patch_cp: 1024`.

Si vous avez **moins de VRAM**, ajustez ainsi:
- **16GB GPU**: `batch_size: 4` + `accum_steps: 6` (batch effectif = 24)
- **12GB GPU**: `batch_size: 3` + `accum_steps: 8` (batch effectif = 24)
- **8GB GPU**: réduire `base_ch: 96 → 64` ou `patch_cp: 1024 → 512`

## Configs Override (Ancienne méthode)

Les fichiers `loss_*.yaml` sont des overrides à utiliser avec `configs/base.yaml`:

```bash
python train.py --config configs/base.yaml --override configs/loss_balanced.yaml --run_name test
```

Mais les configs standalone `run_*.yaml` sont **recommandées** car elles sont complètes et évitent toute confusion.

## Structure de Sortie

Chaque run crée un dossier sous `runs/`:
```
runs/
└── 20260304_143022__base96-balancedlog__loss=BalancedLogSpectralLoss__seed=44__sha=a1b2c3d4/
    ├── config_used.yaml       # Config exacte utilisée
    ├── git_sha.txt            # Commit Git
    ├── git_diff.patch         # Diff Git
    ├── meta.json              # Metadata (date, hostname, versions)
    ├── TensorBoard/           # Logs TensorBoard
    ├── checkpoints/
    │   ├── latest.pth         # Dernier checkpoint (écrasé chaque epoch)
    │   └── best.pth           # Meilleur checkpoint (val_loss)
    └── training.log           # Logs texte
```

## Métriques à Comparer

- **`val_loss`** - Metric principal
- **`val_psnr`** - Qualité pixel
- **`val_ssim`** - Qualité structurelle
- **`val_mae_cp`** - Erreur par control point
- **`val_corr_cp`** - Corrélation par CP
- **`val_gradw`** / **`val_gradcp`** - Préservation des gradients
- **`val_fluence`** - Conservation de la fluence totale

## Notes

- Checkpoints: uniquement `latest.pth` (rolling) et `best.pth` (val_loss min)
- Plus de `checkpoint_epoch_X.pth` pour économiser l'espace disque
- `save_epoch_freq: 5` présent dans config mais n'est plus utilisé pour créer des checkpoints numérotés

