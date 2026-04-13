## 🚀 Guide de Référence Rapide - Entraînement Sianogram

### ✅ Ce qui a été fait aujourd'hui

1. **Simplification du code** (branche `simplification_model`)
   - Suppression des doublons et code mort
   - Modernisation de la structure
   - Fichier principal: `model_simplified.py`

2. **Amélioration des visualisations** 
   - ❌ AVANT: 4 panels (GT, Pred, Signed Error, **Abs Error MAE**)
   - ✅ APRÈS: 4 panels (**X RGB**, GT, Pred, Signed Error)
   - Les 3 premiers canaux DRR sont affichés en RGB (plus informatif!)

3. **Test de l'entraînement**
   - Dataset: `/mnt/LeGrosDisque/Julien/sianogramme/JUJU`
   - 610/668 patients chargés (91.3%)
   - Train: 163 batches, Val: 61 batches

---

### 📁 Fichiers Importants

```
model_simplified.py          # Modèle principal (simplifié)
quick_test_juju.py          # Script de test rapide (5 epochs)
check_training_progress.py  # Vérifier la progression
show_latest_visual.py       # Voir la dernière visualisation

# Documentation
ANALYSE_CHARGEMENT_DATASET.md    # Analyse détaillée du chargement
ANALYSE_EXCLUSIONS_PATIENTS.md   # Pourquoi 57 patients sont exclus
SUMMARY_RGB_VISUALIZATION.md     # Résumé des visualisations RGB
MODIFICATIONS_VISUALS.md         # Détails des modifications

# Scripts de diagnostic
diagnostic_patients_exclus.py    # Analyser les patients sans ptv_channels.json
```

---

### 🎯 Lancer un Entraînement

#### Test Rapide (5 epochs)
```bash
python quick_test_juju.py
```

#### Entraînement Complet
```bash
python main.py --config config.yaml
```

---

### 📊 Voir les Résultats

#### Vérifier la progression
```bash
python check_training_progress.py
```

#### TensorBoard
```bash
tensorboard --logdir checkpoints/QUICK_TEST_JUJU/TensorBoard
# Ouvrir: http://localhost:6006
```

#### Voir la dernière visualisation
```bash
python show_latest_visual.py
```

#### Voir directement les images
```bash
eog checkpoints/QUICK_TEST_JUJU/TensorBoard/*/train_visuals/*.png
```

---

### 📈 Comprendre les Logs

#### ✅ Messages Normaux (OK)
```
[INFO] PTV ptv_br absent -> dose=0, present=0
[INFO] PTV ptv_ri absent -> dose=0, present=0
[INFO] PTV ptv_hr absent -> dose=0, present=0
```
→ Ces PTVs sont optionnels, leur absence est gérée

```
[WARN] cp_dur_sec_mean hors [0,1] après norm -> clip. v=0.280 -> -0.032
```
→ Clipping automatique appliqué (4% des cas)

#### ❌ Messages à Surveiller
```
[SKIP:XXXXX] ptv_channels.json introuvable
```
→ Patient exclu (normal si <10% des cas)

```
RuntimeError: CUDA out of memory
```
→ Réduire batch_size dans la config

---

### 🎨 Nouvelles Visualisations RGB

Chaque image de visualisation contient 4 panels:

```
┌─────────────┬───────────────┬─────────────┬───────────────┐
│             │               │             │               │
│  X (RGB)    │ Ground Truth  │ Prediction  │ Signed Error  │
│  Ch 0,1,2   │               │             │               │
│             │               │             │               │
└─────────────┴───────────────┴─────────────┴───────────────┘
```

**Interprétation**:
- **X (RGB)**: Les DRRs d'entrée (3 angles en couleur)
  - Rouge = Canal 0
  - Vert = Canal 1
  - Bleu = Canal 2
  
- **Ground Truth**: Le sinogramme réel (vérité terrain)

- **Prediction**: Le sinogramme généré par le modèle

- **Signed Error**: La différence (pred - truth)
  - Bleu = sous-estimation
  - Rouge = surestimation
  - Blanc = parfait

---

### 🔧 Configuration Actuelle

```yaml
Dataset:
  Path: /mnt/LeGrosDisque/Julien/sianogramme/JUJU
  Patients: 610/668 (91.3%)
  Train/Val: 163/61 batches
  Batch size: 4

Modèle:
  Type: UNet
  Paramètres: 4.99M
  Base channels: 48
  Depth: 3
  
Training:
  Learning rate: 8e-5
  AMP: BFloat16
  Epochs: 5 (test)
  Device: CUDA
```

---

### 📝 Checklist Avant l'Entraînement

- [x] Dataset chargé (>90% patients)
- [x] Train/Val split correct (~70/30)
- [x] Visualisations RGB activées
- [x] CUDA disponible
- [x] AMP BFloat16 activé
- [x] TensorBoard configuré
- [x] Checkpoints configurés

---

### 🆘 Dépannage Rapide

| Problème | Solution |
|----------|----------|
| CUDA OOM | Réduire `batch_size` |
| Pas de visualisations | Vérifier `visual_epoch_train` et `visual_batch_train` |
| Dataset trop petit | Vérifier les fichiers manquants (ptv_json, X_montage) |
| Perte NaN | Réduire `learning_rate` |

---

### 📞 Fichiers de Log

```
checkpoints/QUICK_TEST_JUJU/
├── training.log          # Log texte complet
├── TensorBoard/          # Métriques et visualisations
│   └── YYYYMMDD-HHMMSS/
│       ├── events.out.*  # Fichier TensorBoard
│       ├── train_visuals/
│       └── val_visuals/
└── checkpoints/          # Modèles sauvegardés
    ├── best_model.pth
    └── epoch_*.pth
```

---

**Date**: 2026-03-04  
**Branche**: `simplification_model`  
**Status**: ✅ Prêt pour l'entraînement!

