# Campagne expérimentale — Guide de lancement (Oussama)

**Config de base** : `configs/run_halo_patch_baseline.yaml`
**Répertoire overrides** : `configs/campaign_oussama/`
**Script tout-en-un** : `configs/campaign_oussama/launch_campaign.sh`
**Tableau de suivi** : `configs/campaign_oussama/tableau_runs.md`

---

## Démarrage rapide

```bash
# Activer l'environnement
conda activate Sinogramme
cd /path/to/sianogram

# Lancer un essai individuel
bash configs/campaign_oussama/launch_campaign.sh H3

# Lancer toute la phase 1 en séquence
bash configs/campaign_oussama/launch_campaign.sh phase1

# Liste des cibles disponibles
bash configs/campaign_oussama/launch_campaign.sh
```

---

## GPU — Sélection du device

Le paramètre `device: cuda` est défini dans `_base_campaign.yaml`.
Si CUDA n'est pas disponible, le code bascule automatiquement sur CPU.

```bash
# GPU par défaut (cuda:0)
bash configs/campaign_oussama/launch_campaign.sh H3

# Forcer un GPU spécifique (méthode recommandée)
CUDA_VISIBLE_DEVICES=1 bash configs/campaign_oussama/launch_campaign.sh H3

# Ou éditer device dans _base_campaign.yaml :
#   device: cuda:1   ← second GPU
#   device: cpu      ← CPU uniquement (debug)
```

---

## Principe de composition des overrides

```
python train.py \
  --config configs/run_halo_patch_baseline.yaml \     ← base complète (figée)
  --override configs/campaign_oussama/_base_campaign.yaml \   ← seed/split/refiner/loss
  --override configs/campaign_oussama/_screening.yaml \       ← durée + visuels
  --override configs/campaign_oussama/phase1_H3.yaml \        ← paramètre testé
  --run_name camp_H3_screen_s44
```

**Règle** : le dernier override l'emporte. Ne jamais réordonner les 3 premiers.

### Fichiers de durée (et paramètres de visualisation)

| Fichier | Epochs | visual_epoch_train | visual_epoch_val | visual_batch_train | visual_batch_val |
|---|---|---|---|---|---|
| `_screening.yaml` | 70 | 10 | 5 | 999 (idx=0 seul.) | 999 (idx=0 seul.) |
| `_medium.yaml` | 140 | 5 | 5 | 100 | 50 |
| `_confirmation.yaml` | 200 | 5 | 2 | 50 | 10 |

> `visual_batch=999` → un seul snapshot par epoch (uniquement idx=0).
> Mettre `visual_batch=0` pour **désactiver complètement** les visuels.

### Seeds pour la confirmation

```bash
--seed 44    # seed A (défaut campagne)
--seed 42    # seed B
--seed 123   # seed C
```

---

## Phase 1 — Géométrie halo (4 runs)

> Figer : `use_patch_cp_head=false`, loss defaults, `LR=8e-5`
> Variable unique : géométrie du patch (patch_in / patch_out / halo).

```bash
# ── H1 ─── Contexte petit — patch_in=512, patch_out=384, halo=64
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H1.yaml \
  --run_name camp_H1_screen_s44

# ── H2 ─── Même cœur (512), halo réduit — patch_in=640, patch_out=512, halo=64
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H2.yaml \
  --run_name camp_H2_screen_s44

# ── H3 ─── BASELINE ACTUEL ★ — patch_in=768, patch_out=512, halo=128
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --run_name camp_H3_screen_s44

# ── H4 ─── Contexte large — patch_in=1024, patch_out=768, halo=128
# ⚠ batch_size=4, accum=3 → effective=12 (défini dans phase1_H4.yaml)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H4.yaml \
  --run_name camp_H4_screen_s44
```

**Ou en une commande :**
```bash
bash configs/campaign_oussama/launch_campaign.sh phase1
```

---

## Phase 2 — Tête locale PatchCPHead1D (2 à 4 runs)

> **Avant de lancer** : éditer `H_BEST` dans `launch_campaign.sh` avec le gagnant Phase 1.
> Figer : géométrie halo gagnante, loss defaults, `LR=8e-5`

```bash
# ── P0 ─── Pas de tête locale (référence)
# Remplacer phase1_H3.yaml par le fichier gagnant de Phase 1
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \   # ← H_BEST
  --override configs/campaign_oussama/phase2_P0.yaml \
  --run_name camp_H3_P0_screen_s44

# ── P1 ─── Tête locale pleine (hidden=128, layers=3)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \   # ← H_BEST
  --override configs/campaign_oussama/phase2_P1.yaml \
  --run_name camp_H3_P1_screen_s44

# ── P2 ─── (si P1 > P0) Tête locale petite (hidden=64, layers=2)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \   # ← H_BEST
  --override configs/campaign_oussama/phase2_P2.yaml \
  --run_name camp_H3_P2_screen_s44

# ── P3 ─── (si P1 > P0) Tête locale intermédiaire (hidden=128, layers=2)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \   # ← H_BEST
  --override configs/campaign_oussama/phase2_P3.yaml \
  --run_name camp_H3_P3_screen_s44
```

**Via script (avec H_BEST mis à jour) :**
```bash
# Éditer H_BEST en haut de launch_campaign.sh, puis :
bash configs/campaign_oussama/launch_campaign.sh P0
bash configs/campaign_oussama/launch_campaign.sh P1
# Si P1 gagne :
bash configs/campaign_oussama/launch_campaign.sh P2
bash configs/campaign_oussama/launch_campaign.sh P3
```

---

## Phase 3 — Loss (5 runs + 1 alternatif)

> **Avant de lancer** : éditer `H_BEST` et `P_BEST` dans `launch_campaign.sh`.
> Figer : géométrie halo + patch head, `LR=8e-5`

```bash
# Remplacer phase1_H3.yaml et phase2_P0.yaml par les gagnants respectifs.

# ── L1 ─── Baseline loss (référence)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \   # ← H_BEST
  --override configs/campaign_oussama/phase2_P0.yaml \   # ← P_BEST
  --override configs/campaign_oussama/phase3_L1.yaml \
  --run_name camp_H3_P0_L1_screen_s44

# ── L2 ─── apply_sigmoid=true
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L2.yaml \
  --run_name camp_H3_P0_L2_screen_s44

# ── L3 ─── lambda_fp=8 (fond plus pénalisé)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L3.yaml \
  --run_name camp_H3_P0_L3_screen_s44

# ── L4 ─── lambda_grad=2 (structure inter-lames renforcée)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L4.yaml \
  --run_name camp_H3_P0_L4_screen_s44

# ── L5 ─── lambda_high=3 (pics renforcés)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L5.yaml \
  --run_name camp_H3_P0_L5_screen_s44

# ── Lalt ─── BalancedLogSpectralLoss (DERNIER RECOURS)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L_alt_balanced.yaml \
  --run_name camp_H3_P0_Lalt_screen_s44
```

**Via script :**
```bash
bash configs/campaign_oussama/launch_campaign.sh phase3
```

---

## Phase 4 — Learning rate (3 runs)

> **Avant de lancer** : éditer `H_BEST`, `P_BEST`, `L_BEST` dans `launch_campaign.sh`.

```bash
# ── O1 ─── LR baseline 8e-5 (référence)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \   # ← H_BEST
  --override configs/campaign_oussama/phase2_P0.yaml \   # ← P_BEST
  --override configs/campaign_oussama/phase3_L1.yaml \   # ← L_BEST
  --override configs/campaign_oussama/phase4_O1.yaml \
  --run_name camp_FINAL_O1_screen_s44

# ── O2 ─── LR=4e-5 (divisé par 2)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L1.yaml \
  --override configs/campaign_oussama/phase4_O2.yaml \
  --run_name camp_FINAL_O2_screen_s44

# ── O3 ─── LR=1.6e-4 (multiplié par 2)
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_screening.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L1.yaml \
  --override configs/campaign_oussama/phase4_O3.yaml \
  --run_name camp_FINAL_O3_screen_s44
```

---

## Phase 5 — Confirmation (top 2 finalistes × 3 seeds)

> **Avant de lancer** : éditer `H_BEST`, `P_BEST`, `L_BEST`, `O_BEST`.
> Run complet 200 epochs. Répéter pour le 2e finaliste.

```bash
# ── Finaliste 1 — seed A
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_confirmation.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \   # ← H_BEST
  --override configs/campaign_oussama/phase2_P0.yaml \   # ← P_BEST
  --override configs/campaign_oussama/phase3_L1.yaml \   # ← L_BEST
  --override configs/campaign_oussama/phase4_O1.yaml \   # ← O_BEST
  --seed 44 --run_name camp_FINAL1_confirm_s44

# ── Finaliste 1 — seed B
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_confirmation.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L1.yaml \
  --override configs/campaign_oussama/phase4_O1.yaml \
  --seed 42 --run_name camp_FINAL1_confirm_s42

# ── Finaliste 1 — seed C
python train.py \
  --config configs/run_halo_patch_baseline.yaml \
  --override configs/campaign_oussama/_base_campaign.yaml \
  --override configs/campaign_oussama/_confirmation.yaml \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --override configs/campaign_oussama/phase3_L1.yaml \
  --override configs/campaign_oussama/phase4_O1.yaml \
  --seed 123 --run_name camp_FINAL1_confirm_s123
```

**Via script :**
```bash
bash configs/campaign_oussama/launch_campaign.sh confirm
```

---

## Critères de rejet immédiat

| Critère | Action |
|---|---|
| `NullXLeakMax` régresse franchement | Rejet direct |
| `BackgroundLeakP99` régresse franchement | Rejet direct |
| Bruit parasite évident sur les visuels | Rejet direct |
| Coutures visibles malgré val_loss correcte | Rejet direct |

---

## Récapitulatif des fichiers

```
configs/campaign_oussama/
├── launch_campaign.sh           ← script bash avec TOUTES les commandes
├── _base_campaign.yaml          ← seed=44, device=cuda, refiner=off, loss defaults
├── _screening.yaml              ← 70 epochs, visuels minimaux
├── _medium.yaml                 ← 140 epochs, visuels modérés
├── _confirmation.yaml           ← 200 epochs, visuels complets
│
├── phase1_H1.yaml               ← 512/384/64
├── phase1_H2.yaml               ← 640/512/64
├── phase1_H3.yaml               ← 768/512/128 ★ baseline
├── phase1_H4.yaml               ← 1024/768/128 ⚠ mémoire
│
├── phase2_P0.yaml               ← patch_head=false
├── phase2_P1.yaml               ← hidden=128, layers=3
├── phase2_P2.yaml               ← hidden=64,  layers=2
├── phase2_P3.yaml               ← hidden=128, layers=2
│
├── phase3_L1.yaml               ← GranularSino baseline
├── phase3_L2.yaml               ← sigmoid=true
├── phase3_L3.yaml               ← λ_fp=8
├── phase3_L4.yaml               ← λ_grad=2
├── phase3_L5.yaml               ← λ_high=3
├── phase3_L_alt_balanced.yaml   ← BalancedLogSpectral (dernier recours)
│
├── phase4_O1.yaml               ← LR=8e-5
├── phase4_O2.yaml               ← LR=4e-5
├── phase4_O3.yaml               ← LR=1.6e-4
│
├── tableau_runs.md              ← tableau de décision à remplir
└── README.md                    ← ce fichier
```

---

## Vérifications

### Cohérence géométrique (validée automatiquement au lancement)

| Config | patch_in = patch_out + 2×halo | ✓ |
|---|---|---|
| H1 | 512 = 384 + 2×64 | ✓ |
| H2 | 640 = 512 + 2×64 | ✓ |
| H3 | 768 = 512 + 2×128 | ✓ |
| H4 | 1024 = 768 + 2×128 | ✓ |

### Effective batch size identique dans tous les runs

| Config | batch_size | accum_steps | effective |
|---|---|---|---|
| H1, H2, H3 | 6 | 2 | **12** |
| H4 | 4 | 3 | **12** |

### jitter_cp = 32 — uniforme pour toute la campagne

Défini dans `_base_campaign.yaml`. Valeur < halo_min=64 → le halo
n'est jamais entièrement consommé par le jitter, quelle que soit la config.
