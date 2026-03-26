# Comment utiliser launch_campaign.sh

---

## Avant tout — se placer au bon endroit

```bash
conda activate Sinogramme
cd /home/julien/PycharmProjects/sianogram
```

> ⚠ Le script doit **toujours** être lancé depuis la racine du projet.
> Les chemins dans le script sont relatifs (`configs/...`, `python train.py`).

---

## Voir ce qui est disponible

```bash
bash configs/campaign_oussama/launch_campaign.sh
```

Affiche la liste de toutes les cibles. Rien n'est lancé.

---

## Lancer un run individuel

```bash
bash configs/campaign_oussama/launch_campaign.sh H3
```

Lance exactement **un** `python train.py` avec les bons YAML, attend la fin, rend la main.

```bash
# Sur un GPU spécifique (si la machine en a plusieurs)
CUDA_VISIBLE_DEVICES=1 bash configs/campaign_oussama/launch_campaign.sh H3
```

---

## Lancer une phase entière en séquence

```bash
bash configs/campaign_oussama/launch_campaign.sh phase1
# → lance H1, puis H2, puis H3, puis H4 l'un après l'autre
```

> ⚠ Si un run échoue (OOM, erreur Python…), le script **s'arrête immédiatement**.
> Relancer uniquement le run raté : `bash ... H4`

---

## La seule chose à modifier entre deux phases

Après chaque phase, tu regardes `tableau_runs.md`, tu choisis le gagnant, et tu ouvres
`launch_campaign.sh` pour mettre à jour **2 lignes** :

```bash
# Exemple : H4 a gagné la Phase 1
# Avant :
H_BEST="configs/campaign_oussama/phase1_H3.yaml"
H_BEST_TAG="H3"

# Après :
H_BEST="configs/campaign_oussama/phase1_H4.yaml"
H_BEST_TAG="H4"
```

Sauvegarder. C'est tout. Les commandes des phases suivantes utilisent automatiquement
la nouvelle valeur sans rien toucher d'autre.

Il y a **4 variables** au total, une par phase :

| Variable | Modifier après | Exemple |
|---|---|---|
| `H_BEST` / `H_BEST_TAG` | Phase 1 | `phase1_H4.yaml` / `"H4"` |
| `P_BEST` / `P_BEST_TAG` | Phase 2 | `phase2_P1.yaml` / `"P1"` |
| `L_BEST` / `L_BEST_TAG` | Phase 3 | `phase3_L3.yaml` / `"L3"` |
| `O_BEST` / `O_BEST_TAG` | Phase 4 | `phase4_O2.yaml` / `"O2"` |

---

## Le déroulé complet de la campagne

```bash
# ── PHASE 1 — Géométrie halo ──────────────────────────────────────────────
bash configs/campaign_oussama/launch_campaign.sh phase1
# Lance : H1, H2, H3, H4 (en séquence)

# → Remplir tableau_runs.md Phase 1
# → Choisir le gagnant (ex: H4)
# → Éditer launch_campaign.sh : H_BEST = phase1_H4.yaml, H_BEST_TAG = "H4"


# ── PHASE 2 — Tête locale ─────────────────────────────────────────────────
bash configs/campaign_oussama/launch_campaign.sh P0
bash configs/campaign_oussama/launch_campaign.sh P1

# → Si P1 gagne sur P0, affiner :
bash configs/campaign_oussama/launch_campaign.sh P2
bash configs/campaign_oussama/launch_campaign.sh P3

# → Remplir tableau_runs.md Phase 2
# → Éditer launch_campaign.sh : P_BEST / P_BEST_TAG


# ── PHASE 3 — Loss ────────────────────────────────────────────────────────
bash configs/campaign_oussama/launch_campaign.sh phase3
# Lance : L1, L2, L3, L4, L5 (en séquence)

# → Remplir tableau_runs.md Phase 3
# → Éditer launch_campaign.sh : L_BEST / L_BEST_TAG

# Si nécessaire (dernier recours) :
bash configs/campaign_oussama/launch_campaign.sh Lalt


# ── PHASE 4 — Learning rate ───────────────────────────────────────────────
bash configs/campaign_oussama/launch_campaign.sh phase4
# Lance : O1, O2, O3 (en séquence)

# → Remplir tableau_runs.md Phase 4
# → Éditer launch_campaign.sh : O_BEST / O_BEST_TAG


# ── PHASE 5 — Confirmation ────────────────────────────────────────────────
bash configs/campaign_oussama/launch_campaign.sh confirm
# Lance : 3 seeds × run complet 200 epochs (finaliste 1)
# Pour le finaliste 2 : lancer les 3 seeds individuellement avec les bons overrides
```

---

## Toutes les cibles disponibles

| Cible | Ce que ça lance |
|---|---|
| `phase1` | H1 + H2 + H3 + H4 en séquence |
| `H1` `H2` `H3` `H4` | Un seul run Phase 1 |
| `phase2` | P0 + P1 en séquence |
| `P0` `P1` `P2` `P3` | Un seul run Phase 2 |
| `phase3` | L1 + L2 + L3 + L4 + L5 en séquence |
| `L1` `L2` `L3` `L4` `L5` `Lalt` | Un seul run Phase 3 |
| `phase4` | O1 + O2 + O3 en séquence |
| `O1` `O2` `O3` | Un seul run Phase 4 |
| `confirm` | 3 seeds × run complet (finaliste 1) |
| `confirm_s44` `confirm_s42` `confirm_s123` | Un seul run de confirmation |

---

## En résumé

```
1. conda activate + cd racine projet
2. bash ... <CIBLE>
3. Regarder les métriques dans tableau_runs.md
4. Éditer 2 lignes dans launch_campaign.sh
5. Répéter à la phase suivante
```

