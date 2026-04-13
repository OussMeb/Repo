# 🚀 Guide de Lancement des Entraînements

Guide pratique pour lancer des entraînements, modifier des paramètres et monitorer les résultats.

---

## 📋 Lancer une Run

### Lancer une run simple
```bash
python train.py \
  --config configs/base.yaml \
  --run_name my_first_run
```

### Lancer une run de production (BalancedLogSpectralLoss)
```bash
python train.py \
  --config configs/run_balancedlog.yaml \
  --run_name prod_baseline
```

---

## ⚡ Guide Rapide Inference

### Inference depuis une run existante (checkpoint `best`)
```bash
python inference.py \
  --resume-from runs/20260305_160200__thresholds__loss=TomoSinoStrictZeroLoss__seed=44__sha=3282a6d0 \
  --checkpoint-type best \
  --out-dir inference_outputs/my_run_best
```

### Inference depuis le checkpoint `latest`
```bash
python inference.py \
  --resume-from runs/20260305_160200__thresholds__loss=TomoSinoStrictZeroLoss__seed=44__sha=3282a6d0 \
  --checkpoint-type latest \
  --out-dir inference_outputs/my_run_latest
```

### Inference + injection RTPLAN directe
```bash
python inference.py \
  --resume-from runs/20260305_160200__thresholds__loss=TomoSinoStrictZeroLoss__seed=44__sha=3282a6d0 \
  --checkpoint-type best \
  --inject-rtplan \
  --dicom-root /mnt/LeGrosDisque/Julien/sianogramme/dicom_data \
  --out-dir inference_outputs/my_run_rtplan
```

### Defaults RTPLAN actuels (injecteur)
- `RTPlanLabel/RTPlanName` renommés en `sIAnogram_YYYYMMDD`
- `TreatmentMachineName` fixé à `Rdx_1_dble_calc`
- Seuils d'ouverture: `low=20ms`, `high=0ms` (désactivé)
- Astuce: mettre `--low-thresh 0 --high-thresh 0` si tu veux désactiver tout seuillage

---

## 🎯 Modifier les Paramètres d'une Run

### Avec un fichier override (recommandé pour garder une trace)

Créer `configs/my_experiment.yaml` avec les changements :
```yaml
training:
  learning_rate: 8e-5
  n_epochs: 150
  scheduler_patience: 5
```

Lancer :
```bash
python train.py \
  --config configs/base.yaml \
  --override configs/my_experiment.yaml \
  --run_name my_experiment
```

### Avec des flags CLI (rapide, pour tester)

```bash
python train.py \
  --config configs/base.yaml \
  --run_name quick_test \
  --lr 5e-5 \
  --epochs 100 \
  --seed 123
```

### Combiner les deux

```bash
python train.py \
  --config configs/base.yaml \
  --override configs/my_experiment.yaml \
  --run_name my_experiment_with_cli_override \
  --lr 4e-5
```

✅ **Les flags CLI (`--lr`, `--epochs`) ont la plus haute priorité**

---

## 🧪 Exemples Courants

### Test Base avec MSELoss
```bash
python train.py \
  --config configs/base.yaml \
  --run_name test_mse
```


### Comparer Plusieurs Learning Rates

```bash
for LR in 5e-5 6e-5 8e-5; do
  cat > /tmp/lr_${LR}.yaml << EOF
training:
  learning_rate: ${LR}
  n_epochs: 50   # Test rapide
EOF
  
  python train.py \
    --config configs/base.yaml \
    --override /tmp/lr_${LR}.yaml \
    --run_name lr_${LR} &
done
wait
```

### Comparer Les 4 Loss Functions

```bash
python train.py --config configs/run_balancedlog.yaml --run_name loss_balanced &
python train.py --config configs/run_sparsefocal.yaml --run_name loss_focal &
python train.py --config configs/run_sparsesino.yaml --run_name loss_sino &
python train.py --config configs/run_gigaultimate.yaml --run_name loss_giga &
wait
```

---

## 🔁 Reprendre depuis un Checkpoint

#### Cas 1: Reprendre simplement (on continue juste)

```bash
python train.py --resume-from runs/20260304_140849__newmodel* --run_name resume_best
```

✅ **Qu'est-ce qui se passe** :
- Trouve automatiquement le dossier `runs/20260304_140849__newmodel*`
- Charge `config_used.yaml` (la config exacte de l'ancienne run)
- Charge `checkpoints/best.pth` (le meilleur modèle)
- Crée une **nouvelle run** avec le nom `resume_best`
- Continue l'entraînement à partir du checkpoint

#### Cas 2: Reprendre et tester d'autres paramètres (le plus courant!)

**Exemple: Reprendre mais diminuer le learning rate**

```bash
python train.py --resume-from runs/20260304_* \
  --run_name resume_lower_lr \
  --lr 4e-5
```

**Exemple: Reprendre et ajouter plus d'epochs avec visualisation**

```bash
python train.py --resume-from runs/20260304_* \
  --run_name resume_more_epochs \
  --epochs 300 \
  --vis-val 1 \
  --vis-train 1
```

**Exemple: Reprendre avec plusieurs changements**

```bash
python train.py --resume-from runs/20260304_* \
  --run_name resume_exp_changes \
  --lr 5e-5 \
  --epochs 250 \
  --vis-val 1
```

✅ **Priorité des changements** (du plus faible au plus fort) :
1. Config de base (`config_used.yaml`) 
2. Fichiers `--override` (si tu les ajoutes)
3. Flags CLI (`--lr`, `--epochs`, `--vis-val`) ← **plus haute priorité**

#### Cas 3: Reprendre depuis le "latest" au lieu du "best"

```bash
# Par défaut c'est best.pth
python train.py --resume-from runs/20260304_* --run_name resume_best

# Pour reprendre depuis latest.pth
python train.py --resume-from runs/20260304_* \
  --checkpoint-type latest \
  --run_name resume_latest
```

**Quand utiliser `latest` vs `best`** :
- `best` ✅ (défaut) = meilleur modèle selon validation loss (recommandé)
- `latest` = dernier modèle sauvegardé (si tu veux continuer l'entraînement sans interruption)

#### Cas 4: Reprendre avec override file (changements complexes)

```bash
# Créer un fichier de changements
cat > configs/resume_changes.yaml << EOF
training:
  learning_rate: 4e-5
  scheduler_patience: 10
  n_epochs: 300
EOF

# Relancer avec ce fichier
python train.py --resume-from runs/20260304_* \
  --override configs/resume_changes.yaml \
  --run_name resume_advanced
```

✅ **Quand utiliser `--override`** :
- Pour des changements complexes (plusieurs paramètres)
- Pour documenter les changements dans un fichier réutilisable
- Sinon, utilise juste `--lr 5e-5 --epochs 250` directement

---


## 📊 Organisation des Résultats

Chaque run crée automatiquement:
```
runs/
└── YYYYMMDD_HHMMSS__run_name__loss=LossName__seed=44__sha=xxx/
    ├── config_used.yaml           # Config exacte utilisée
    ├── git_sha.txt                # Commit Git
    ├── training.log               # Logs texte
    ├── checkpoints/
    │   ├── best.pth              # Meilleur modèle
    │   └── latest.pth            # Dernier modèle
    ├── TensorBoard/
    │   └── events.out.tfevents.*
    ├── train_visuals/            # Images training
    └── val_visuals/              # Images validation
```

### Monitorer en Temps Réel

#### TensorBoard (Recommandé - compare toutes les runs)

**Lancer TensorBoard** :
```bash
tensorboard --logdir runs/
```

**Accéder au dashboard** :
- Ouvre ton navigateur et va sur : **http://localhost:6006**
- Tu verras tous tes runs avec leurs métriques en temps réel
- Compare facilement plusieurs runs côte à côte

**Shortcuts utiles** :
- **SCALARS** : loss, accuracy, learning rate (ce que tu veux voir le plus)
- **IMAGES** : visualisations train/val (tes prédictions vs ground truth)
- **DISTRIBUTIONS** : histogrammes des poids du modèle

**Arrêter TensorBoard** :
```bash
# Ctrl+C dans le terminal où tu l'as lancé
```

#### Logs texte (pour debug rapide)

```bash
# Voir les logs de la run en cours
tail -f runs/YYYYMMDD_HHMMSS__run_name__loss=Loss__seed=44__sha=xxx/training.log

# Ou tous les logs
tail -f runs/*/training.log
```

#### GPU (vérifier que ça utilise bien la GPU)
```bash
watch -n 1 nvidia-smi
```

---

## 📝 Fichiers de Configuration

### Structure de base.yaml
```yaml
# Données
data:
  batch_size: 12
  patch_cp: 512
  cp_height: 12

# Modèle
model:
  base_ch: 128
  depth: 3
  use_transformer: false

# Entraînement
training:
  learning_rate: 6e-5
  n_epochs: 200
  
# Loss
loss:
  name: MSELoss              # Par défaut
```

### Loss Functions Disponibles

| Config | Loss Function | Usage |
|--------|---------------|-------|
| `base.yaml` | MSELoss | Test rapide |
| `run_balancedlog.yaml` | BalancedLogSpectralLoss | Recommandé - production |
| `run_sparsefocal.yaml` | SparseFocalSpectralLoss | Expérimental |
| `run_sparsesino.yaml` | SparseSinoLoss | Expérimental |
| `run_gigaultimate.yaml` | GigaUltimateLoss | Expérimental |

---

## 🔄 Lancer en Arrière-Plan

### Avec `nohup` (Simple)
```bash
nohup python train.py --config configs/base.yaml --run_name bg_test > train.log 2>&1 &
```

### Avec `screen` (Meilleur pour long terme)
```bash
screen -S training
python train.py --config configs/base.yaml --run_name my_run
# Détacher: Ctrl+A puis D
# Réattacher: screen -r training
```

### Avec `tmux` (Puissant)
```bash
tmux new -s training -d "python train.py --config configs/base.yaml --run_name my_run"
tmux attach -t training
```

---

## 🛠️ Troubleshooting

### Out of Memory
```yaml
# Baisser le batch size
data:
  batch_size: 6  # Au lieu de 12

# Ou réduire les canaux du modèle
model:
  base_ch: 64    # Au lieu de 128
```

### Run Bloquée
```bash
# Vérifier les process
ps aux | grep train.py

# Voir les logs
tail -f runs/*/training.log

# Tuer si nécessaire
pkill -f train.py
```

### Loss = NaN
```yaml
# Baisser le learning rate
training:
  learning_rate: 3e-5  # Au lieu de 6e-5
```

---

## 📚 Ressources

- **Configs détaillées** : `configs/README.md`
- **Architecture exp branch** : `documentation/NETWORK_EXP_CHANGES.md`

---

**Dernière mise à jour** : 4 mars 2026

