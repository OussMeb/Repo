# 🚀 Guide de Lancement - Dataset JUJU

## Votre Dataset

**Chemin**: `/mnt/LeGrosDisque/Julien/sianogramme/JUJU`

## 📋 Prérequis

Avant de lancer l'entraînement, vérifiez que vous avez:

```bash
# 1. Activer votre environnement conda
conda activate Sinogramme

# 2. Vérifier que le dataset existe
ls /mnt/LeGrosDisque/Julien/sianogramme/JUJU

# 3. Vérifier que CUDA fonctionne
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## 🧪 Option 1: Test Rapide (RECOMMANDÉ)

**Avant un long entraînement, testez que tout fonctionne:**

```bash
cd /home/julien/PycharmProjects/sianogram
python quick_test_juju.py
```

**Ce que fait ce test:**
- ✅ Charge votre dataset JUJU
- ✅ Crée un petit modèle (rapide)
- ✅ Entraîne pendant 5 epochs (~5-15 min)
- ✅ Valide que tout fonctionne

**Si le test réussit**, passez à l'entraînement complet!

## 🏃 Option 2: Entraînement Complet

```bash
cd /home/julien/PycharmProjects/sianogram
python launch_training_juju.py
```

**Configuration par défaut:**
- 📊 **Epochs**: 200
- 📦 **Batch size**: 6 (effectif: 12 avec accumulation)
- 🧠 **Modèle**: U-Net base_ch=64, depth=4
- ⚡ **AMP**: Activé (BFloat16)
- 📈 **EMA**: Activé
- 🎨 **Augmentation**: Activée

**Durée estimée**: Selon votre GPU et la taille du dataset
- RTX 3090/4090: ~10-20h pour 200 epochs
- RTX 3080: ~15-30h
- Plus ancien: 30-50h

## 🛠️ Personnalisation

### Modifier les Paramètres

Éditez `launch_training_juju.py`, fonction `build_config_juju()`:

```python
# Exemples de modifications courantes:

# 1. RÉDUIRE LA MÉMOIRE GPU
data=DataConfig(
    batch_size=4,        # Au lieu de 6
    patch_cp=256,        # Au lieu de 512
    ...
)

# 2. AUGMENTER LA VITESSE (sacrifice précision)
model=ModelConfig(
    base_ch=48,          # Au lieu de 64
    depth=3,             # Au lieu de 4
    ...
)

# 3. ENTRAÎNEMENT PLUS COURT
training=TrainingConfig(
    n_epochs=50,         # Au lieu de 200
    ...
)

# 4. CHANGER LE LEARNING RATE
training=TrainingConfig(
    learning_rate=5e-5,  # Au lieu de 8e-5
    ...
)
```

## 📊 Monitoring Pendant l'Entraînement

### 1. Logs en Direct

```bash
# Dans un autre terminal
tail -f checkpoints/juju_unet_bs6_lr8e-5/training.log
```

### 2. TensorBoard

```bash
# Lancer TensorBoard
tensorboard --logdir checkpoints/juju_unet_bs6_lr8e-5/TensorBoard

# Puis ouvrir dans le navigateur:
# http://localhost:6006
```

**Métriques disponibles:**
- Loss (train/val)
- PSNR, SSIM
- MAE per CP
- Pearson correlation
- Gradient MAE
- Fluence MAE

### 3. Visualisations

Les images sont sauvegardées automatiquement dans:
```
checkpoints/juju_unet_bs6_lr8e-5/TensorBoard/<timestamp>/
├── train_visuals/     # Visualisations entraînement
└── val_visuals/       # Visualisations validation
```

## ⏸️ Arrêter et Reprendre

### Arrêter Proprement

Appuyez sur `Ctrl+C` - le modèle sera sauvegardé automatiquement.

### Reprendre l'Entraînement

Éditez `launch_training_juju.py`:

```python
config = Config(
    # ... même configuration ...
    resume=True,
    checkpoint_path='checkpoints/juju_unet_bs6_lr8e-5/checkpoints/latest.pth'
)
```

Puis relancez:
```bash
python launch_training_juju.py
```

## 💾 Checkpoints

Les checkpoints sont sauvegardés dans:
```
checkpoints/juju_unet_bs6_lr8e-5/checkpoints/
├── best.pth                    # Meilleur modèle (val_loss)
├── latest.pth                  # Dernier checkpoint
└── checkpoint_epoch_X.pth      # Checkpoints tous les 5 epochs
```

## 🐛 Résolution de Problèmes

### Erreur: CUDA Out of Memory

**Solution 1**: Réduire le batch size
```python
data=DataConfig(batch_size=4)  # ou 3, ou 2
```

**Solution 2**: Réduire la taille des patches
```python
data=DataConfig(patch_cp=256)  # au lieu de 512
```

**Solution 3**: Désactiver l'accumulation de gradients
```python
training=TrainingConfig(accum_steps=1)
```

### Erreur: Dataset non trouvé

Vérifiez le chemin:
```bash
ls -la /mnt/LeGrosDisque/Julien/sianogramme/JUJU
```

Si le dataset est ailleurs, modifiez dans `build_config_juju()`:
```python
data=DataConfig(path='/nouveau/chemin/vers/dataset')
```

### Entraînement Trop Lent

**Option 1**: Réduire la taille du modèle
```python
model=ModelConfig(
    base_ch=48,     # au lieu de 64
    depth=3,        # au lieu de 4
)
```

**Option 2**: Augmenter le batch size (si vous avez la RAM)
```python
data=DataConfig(batch_size=8)
```

**Option 3**: Plus de workers
```python
data=DataConfig(num_workers=4)  # au lieu de 2
```

### Loss devient NaN

**Solution 1**: Réduire le learning rate
```python
training=TrainingConfig(learning_rate=5e-5)  # au lieu de 8e-5
```

**Solution 2**: Augmenter le gradient clipping
```python
training=TrainingConfig(clip_grad_norm=1.0)  # au lieu de 0.5
```

## 📈 Analyser les Résultats

### Après l'Entraînement

```python
# Charger le meilleur modèle
from model_simplified import Model, Config

config = ...  # Votre config
model = Model(config)
model.load_checkpoint('checkpoints/juju_unet_bs6_lr8e-5/checkpoints/best.pth')

# Le modèle est prêt pour l'inférence!
```

### Comparer Plusieurs Runs

```bash
# Lancer TensorBoard sur plusieurs expériences
tensorboard --logdir checkpoints/
```

## 💡 Conseils

### Pour un Premier Entraînement

1. **Lancez d'abord le test rapide**: `python quick_test_juju.py`
2. **Vérifiez les visualisations** après quelques epochs
3. **Surveillez la loss** - elle doit diminuer
4. **Patience** - l'entraînement prend du temps!

### Pour Optimiser les Performances

1. **Augmentation des données**: Toujours activée (`augment=True`)
2. **AMP**: Activé pour vitesse (`use_amp=True`)
3. **EMA**: Activé pour stabilité (`use_ema=True`)
4. **Batch size**: Aussi grand que possible (selon GPU)

### Pour Débugger

1. **Visualisations fréquentes**: `visual_batch_train=10`
2. **Petit modèle**: `base_ch=48, depth=3`
3. **Peu d'epochs**: `n_epochs=10`
4. **Pas d'augmentation**: `augment=False`

## 📞 Commandes Utiles

```bash
# Test rapide (5 epochs)
python quick_test_juju.py

# Entraînement complet
python launch_training_juju.py

# Voir les logs
tail -f checkpoints/juju_*/training.log

# TensorBoard
tensorboard --logdir checkpoints/

# Vérifier GPU
nvidia-smi

# Tuer un processus bloqué
pkill -f launch_training_juju.py
```

## 🎯 Checklist Avant de Lancer

- [ ] Dataset existe: `ls /mnt/LeGrosDisque/Julien/sianogramme/JUJU`
- [ ] CUDA fonctionne: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Environnement activé: `conda activate Sinogramme`
- [ ] Test rapide réussi: `python quick_test_juju.py`
- [ ] Configuration personnalisée (si besoin)
- [ ] TensorBoard prêt: `tensorboard --logdir checkpoints/`

## ✅ C'est Parti!

```bash
# 1. Test rapide d'abord
python quick_test_juju.py

# 2. Si OK, entraînement complet
python launch_training_juju.py
```

**Bonne chance avec votre entraînement!** 🚀

---

**Questions fréquentes:**
- **Combien de temps?** 10-30h selon GPU et dataset
- **Puis-je arrêter?** Oui, Ctrl+C puis reprendre avec `resume=True`
- **Out of memory?** Réduire `batch_size` ou `patch_cp`
- **Loss NaN?** Réduire `learning_rate`

---

**Fichiers créés pour vous:**
- `launch_training_juju.py` - Entraînement complet (200 epochs)
- `quick_test_juju.py` - Test rapide (5 epochs)
- Ce guide - Comment tout utiliser

