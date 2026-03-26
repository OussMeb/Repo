# Tableau de runs — Campagne Oussama

> Remplir une ligne par run. Colonne `statut` : **retenu** / **rejeté** / **à revoir**

---

## Phase 1 — Géométrie halo

| run_name | seed | patch_in | patch_out | halo | patch_head | loss | LR | val_loss↓ | SeamScore↓ | MAE_EnergyCP↓ | F1_open↑ | MAE_open↓ | BgLeakP99↓ | NullXLeakMax↓ | commentaire visuel | statut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| camp_H1_screen_s44 | 44 | 512 | 384 | 64 | non | Granular L1 | 8e-5 | | | | | | | | | |
| camp_H2_screen_s44 | 44 | 640 | 512 | 64 | non | Granular L1 | 8e-5 | | | | | | | | | |
| camp_H3_screen_s44 | 44 | 768 | 512 | 128 | non | Granular L1 | 8e-5 | | | | | | | | | |
| camp_H4_screen_s44 | 44 | 1024 | 768 | 128 | non | Granular L1 | 8e-5 | | | | | | | | | |

**Décision Phase 1 :**
- H retenu 1 : `_______`
- H retenu 2 : `_______`
- Critère utilisé : `_______`

---

## Phase 2 — Tête locale PatchCPHead1D

> Géométrie halo fixée à : **H_BEST = `_______`**

| run_name | seed | patch_in | patch_out | halo | patch_head | hidden | layers | loss | LR | val_loss↓ | SeamScore↓ | MAE_EnergyCP↓ | F1_open↑ | MAE_open↓ | BgLeakP99↓ | NullXLeakMax↓ | commentaire visuel | statut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| camp_HBEST_P0_screen_s44 | 44 | ? | ? | ? | non | — | — | Granular L1 | 8e-5 | | | | | | | | | |
| camp_HBEST_P1_screen_s44 | 44 | ? | ? | ? | oui | 128 | 3 | Granular L1 | 8e-5 | | | | | | | | | |
| camp_HBEST_P2_screen_s44 | 44 | ? | ? | ? | oui | 64 | 2 | Granular L1 | 8e-5 | | | | | | | | | |
| camp_HBEST_P3_screen_s44 | 44 | ? | ? | ? | oui | 128 | 2 | Granular L1 | 8e-5 | | | | | | | | | |

**Décision Phase 2 :**
- Patch head retenu : **oui / non** — config : `_______`
- Critère utilisé : `_______`

---

## Phase 3 — Loss

> Géométrie halo fixée à : **H_BEST = `_______`** | Patch head : **`_______`**

| run_name | seed | patch_head | loss | lambda_fp | lambda_grad | lambda_high | sigmoid | LR | val_loss↓ | SeamScore↓ | MAE_EnergyCP↓ | F1_open↑ | MAE_open↓ | BgLeakP99↓ | NullXLeakMax↓ | commentaire visuel | statut |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| camp_HBEST_PBEST_L1_screen_s44 | 44 | ? | Granular | 5.0 | 1.0 | 2.0 | false | 8e-5 | | | | | | | | | |
| camp_HBEST_PBEST_L2_screen_s44 | 44 | ? | Granular | 5.0 | 1.0 | 2.0 | true  | 8e-5 | | | | | | | | | |
| camp_HBEST_PBEST_L3_screen_s44 | 44 | ? | Granular | 8.0 | 1.0 | 2.0 | false | 8e-5 | | | | | | | | | |
| camp_HBEST_PBEST_L4_screen_s44 | 44 | ? | Granular | 5.0 | 2.0 | 2.0 | false | 8e-5 | | | | | | | | | |
| camp_HBEST_PBEST_L5_screen_s44 | 44 | ? | Granular | 5.0 | 1.0 | 3.0 | false | 8e-5 | | | | | | | | | |

**Décision Phase 3 :**
- Loss retenue : `_______`
- Critère utilisé : `_______`

---

## Phase 4 — Learning rate

> Config finale fixée (halo + patch head + loss)

| run_name | seed | LR | val_loss↓ | SeamScore↓ | MAE_EnergyCP↓ | F1_open↑ | MAE_open↓ | BgLeakP99↓ | NullXLeakMax↓ | commentaire visuel | statut |
|---|---|---|---|---|---|---|---|---|---|---|---|
| camp_FINAL_O1_screen_s44 | 44 | 8e-5 | | | | | | | | | |
| camp_FINAL_O2_screen_s44 | 44 | 4e-5 | | | | | | | | | |
| camp_FINAL_O3_screen_s44 | 44 | 1.6e-4 | | | | | | | | | |

**Décision Phase 4 :**
- LR retenu : `_______`

---

## Phase 5 — Confirmation (top 2 finalistes × 3 seeds)

| run_name | seed | config | val_loss↓ | SeamScore↓ | MAE_EnergyCP↓ | F1_open↑ | MAE_open↓ | BgLeakP99↓ | NullXLeakMax↓ | statut |
|---|---|---|---|---|---|---|---|---|---|---|---|
| camp_FINAL1_confirm_s44 | 44 | finaliste 1 | | | | | | | | |
| camp_FINAL1_confirm_s42 | 42 | finaliste 1 | | | | | | | | |
| camp_FINAL1_confirm_s123 | 123 | finaliste 1 | | | | | | | | |
| camp_FINAL2_confirm_s44 | 44 | finaliste 2 | | | | | | | | |
| camp_FINAL2_confirm_s42 | 42 | finaliste 2 | | | | | | | | |
| camp_FINAL2_confirm_s123 | 123 | finaliste 2 | | | | | | | | |

**Gagnant final** : `_______`
Raison : meilleure moyenne + plus faible variance entre seeds.

