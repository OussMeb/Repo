#!/usr/bin/env bash
# =============================================================================
# CAMPAGNE OUSSAMA — Script de lancement complet
# =============================================================================
# Ce script contient TOUTES les commandes de la campagne, organisées par phase.
#
# AVANT DE LANCER :
#   1. Activer l'environnement conda : conda activate Sinogramme
#   2. Se placer à la racine du projet : cd /path/to/sianogram
#   3. Pour un GPU spécifique : export CUDA_VISIBLE_DEVICES=0
#
# USAGE — lancer un essai individuel (recommandé) :
#   bash configs/campaign_oussama/launch_campaign.sh H3
#
# USAGE — lancer toute la phase 1 en séquence :
#   bash configs/campaign_oussama/launch_campaign.sh phase1
#
# PHASES DISPONIBLES : phase1 | phase2 | phase3 | phase4 | confirm
# ESSAIS INDIVIDUELS : H1 H2 H3 H4 | P0 P1 P2 P3 | L1 L2 L3 L4 L5 | O1 O2 O3
#
# NOTE PHASES 2-4 : mettre à jour les variables H_BEST / P_BEST / L_BEST / O_BEST
#                  après chaque phase avant de lancer la phase suivante.
# =============================================================================

set -e  # arrêter en cas d'erreur

# =============================================================================
# VARIABLES À METTRE À JOUR APRÈS CHAQUE PHASE
# =============================================================================
BASE="configs/run_halo_patch_baseline.yaml"
CAMP="configs/campaign_oussama/_base_campaign.yaml"
SCREEN="configs/campaign_oussama/_screening.yaml"
MEDIUM="configs/campaign_oussama/_medium.yaml"
CONFIRM="configs/campaign_oussama/_confirmation.yaml"

# ↓↓↓ METTRE À JOUR APRÈS PHASE 1 ↓↓↓
H_BEST="configs/campaign_oussama/phase1_H1.yaml"   # ex: phase1_H3.yaml ou phase1_H4.yaml
H_BEST_TAG="H1"

# ↓↓↓ METTRE À JOUR APRÈS PHASE 2 ↓↓↓
P_BEST="configs/campaign_oussama/phase2_P1.yaml"   # ex: phase2_P0.yaml ou phase2_P1.yaml
P_BEST_TAG="P0"

# ↓↓↓ METTRE À JOUR APRÈS PHASE 3 ↓↓↓
L_BEST="configs/campaign_oussama/phase3_L1.yaml"   # ex: phase3_L1.yaml ... phase3_L5.yaml
L_BEST_TAG="L1"

# ↓↓↓ METTRE À JOUR APRÈS PHASE 4 ↓↓↓
O_BEST="configs/campaign_oussama/phase4_O1.yaml"   # ex: phase4_O1.yaml ... phase4_O3.yaml
O_BEST_TAG="O1"

# =============================================================================
# PHASE 1 — Géométrie halo (screening 70 epochs, 1 seed)
# =============================================================================

run_H1() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override configs/campaign_oussama/phase1_H1.yaml \
  --run_name camp_H1_screen_s44
}

run_H2() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override configs/campaign_oussama/phase1_H2.yaml \
  --run_name camp_H2_screen_s44
}

run_H3() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override configs/campaign_oussama/phase1_H3.yaml \
  --run_name camp_H3_screen_s44
}

run_H4() {
# ⚠ batch_size réduit à 4 + accum=3 → effective=12 (contenu dans phase1_H4.yaml)
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override configs/campaign_oussama/phase1_H4.yaml \
  --run_name camp_H4_screen_s44
}

run_phase1() {
  echo "===== PHASE 1 : Géométrie halo ====="
  run_H1
  run_H2
  run_H3
  run_H4
}

# =============================================================================
# PHASE 2 — Tête locale PatchCPHead1D (screening, H_BEST figé)
# =============================================================================
# Mettre à jour H_BEST avant de lancer !

run_P0() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override configs/campaign_oussama/phase2_P0.yaml \
  --run_name "camp_${H_BEST_TAG}_P0_screen_s44"
}

run_P1() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override configs/campaign_oussama/phase2_P1.yaml \
  --run_name "camp_${H_BEST_TAG}_P1_screen_s44"
}

run_P2() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override configs/campaign_oussama/phase2_P2.yaml \
  --run_name "camp_${H_BEST_TAG}_P2_screen_s44"
}

run_P3() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override configs/campaign_oussama/phase2_P3.yaml \
  --run_name "camp_${H_BEST_TAG}_P3_screen_s44"
}

run_phase2() {
  echo "===== PHASE 2 : Patch head (H_BEST=${H_BEST_TAG}) ====="
  run_P0
  run_P1
  # Lancer P2 et P3 seulement si P1 > P0 :
  # run_P2
  # run_P3
}

# =============================================================================
# PHASE 3 — Loss (screening, H_BEST + P_BEST figés)
# =============================================================================
# Mettre à jour H_BEST et P_BEST avant de lancer !

run_L1() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override configs/campaign_oussama/phase3_L1.yaml \
  --run_name "camp_${H_BEST_TAG}_${P_BEST_TAG}_L1_screen_s44"
}

run_L2() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override configs/campaign_oussama/phase3_L2.yaml \
  --run_name "camp_${H_BEST_TAG}_${P_BEST_TAG}_L2_screen_s44"
}

run_L3() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override configs/campaign_oussama/phase3_L3.yaml \
  --run_name "camp_${H_BEST_TAG}_${P_BEST_TAG}_L3_screen_s44"
}

run_L4() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override configs/campaign_oussama/phase3_L4.yaml \
  --run_name "camp_${H_BEST_TAG}_${P_BEST_TAG}_L4_screen_s44"
}

run_L5() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override configs/campaign_oussama/phase3_L5.yaml \
  --run_name "camp_${H_BEST_TAG}_${P_BEST_TAG}_L5_screen_s44"
}

run_L_alt() {
# DERNIER RECOURS uniquement — si GranularSino reste frustrante après L1→L5
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override configs/campaign_oussama/phase3_L_alt_balanced.yaml \
  --run_name "camp_${H_BEST_TAG}_${P_BEST_TAG}_Lalt_screen_s44"
}

run_phase3() {
  echo "===== PHASE 3 : Loss (H=${H_BEST_TAG}, P=${P_BEST_TAG}) ====="
  run_L1
  run_L2
  run_L3
  run_L4
  run_L5
}

# =============================================================================
# PHASE 4 — Learning rate (screening, H+P+L figés)
# =============================================================================
# Mettre à jour H_BEST, P_BEST, L_BEST avant de lancer !

run_O1() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override "$L_BEST" \
  --override configs/campaign_oussama/phase4_O1.yaml \
  --run_name "camp_FINAL_O1_screen_s44"
}

run_O2() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override "$L_BEST" \
  --override configs/campaign_oussama/phase4_O2.yaml \
  --run_name "camp_FINAL_O2_screen_s44"
}

run_O3() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$SCREEN" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override "$L_BEST" \
  --override configs/campaign_oussama/phase4_O3.yaml \
  --run_name "camp_FINAL_O3_screen_s44"
}

run_phase4() {
  echo "===== PHASE 4 : Learning rate (H=${H_BEST_TAG}, P=${P_BEST_TAG}, L=${L_BEST_TAG}) ====="
  run_O1
  run_O2
  run_O3
}

# =============================================================================
# PHASE 5 — Confirmation (run complet 200 epochs × 3 seeds)
# =============================================================================
# Mettre à jour H_BEST, P_BEST, L_BEST, O_BEST avant de lancer !
# Lancer le top 2 finalistes — montré ici pour finaliste 1 uniquement.
# Dupliquer les fonctions pour finaliste 2 si nécessaire.

run_confirm_s44() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$CONFIRM" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override "$L_BEST" \
  --override "$O_BEST" \
  --seed 44 \
  --run_name "camp_FINAL1_confirm_s44"
}

run_confirm_s42() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$CONFIRM" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override "$L_BEST" \
  --override "$O_BEST" \
  --seed 42 \
  --run_name "camp_FINAL1_confirm_s42"
}

run_confirm_s123() {
python train.py \
  --config "$BASE" \
  --override "$CAMP" \
  --override "$CONFIRM" \
  --override "$H_BEST" \
  --override "$P_BEST" \
  --override "$L_BEST" \
  --override "$O_BEST" \
  --seed 123 \
  --run_name "camp_FINAL1_confirm_s123"
}

run_confirm() {
  echo "===== PHASE 5 : Confirmation (H=${H_BEST_TAG}, P=${P_BEST_TAG}, L=${L_BEST_TAG}, O=${O_BEST_TAG}) ====="
  run_confirm_s44
  run_confirm_s42
  run_confirm_s123
}

# =============================================================================
# DISPATCHER — sélectionner ce qu'on veut lancer
# =============================================================================
# Usage: bash launch_campaign.sh <cible>
# Cibles disponibles :
#   phase1        → H1 H2 H3 H4 (en séquence)
#   H1 / H2 / H3 / H4
#   phase2        → P0 P1 (P2 P3 en option dans la fonction)
#   P0 / P1 / P2 / P3
#   phase3        → L1 L2 L3 L4 L5
#   L1 / L2 / L3 / L4 / L5 / Lalt
#   phase4        → O1 O2 O3
#   O1 / O2 / O3
#   confirm       → s44 s42 s123 (top 1 finaliste)
#   all_phases    → phase1 → phase2 → phase3 → phase4 (EN SÉQUENCE COMPLÈTE)
# =============================================================================

TARGET="${1:-}"

case "$TARGET" in
  # Phase 1
  phase1) run_phase1 ;;
  H1)     run_H1 ;;
  H2)     run_H2 ;;
  H3)     run_H3 ;;
  H4)     run_H4 ;;

  # Phase 2
  phase2) run_phase2 ;;
  P0)     run_P0 ;;
  P1)     run_P1 ;;
  P2)     run_P2 ;;
  P3)     run_P3 ;;

  # Phase 3
  phase3) run_phase3 ;;
  L1)     run_L1 ;;
  L2)     run_L2 ;;
  L3)     run_L3 ;;
  L4)     run_L4 ;;
  L5)     run_L5 ;;
  Lalt)   run_L_alt ;;

  # Phase 4
  phase4) run_phase4 ;;
  O1)     run_O1 ;;
  O2)     run_O2 ;;
  O3)     run_O3 ;;

  # Confirmation
  confirm)    run_confirm ;;
  confirm_s44)  run_confirm_s44 ;;
  confirm_s42)  run_confirm_s42 ;;
  confirm_s123) run_confirm_s123 ;;

  # Tout enchaîner (à n'utiliser qu'en connaissance de cause)
  all_phases)
    run_phase1
    echo "=== METTRE À JOUR H_BEST AVANT DE CONTINUER ==="
    ;;

  "")
    echo ""
    echo "Usage: bash configs/campaign_oussama/launch_campaign.sh <cible>"
    echo ""
    echo "  Phase 1 (halo)  : phase1 | H1 H2 H3 H4"
    echo "  Phase 2 (head)  : phase2 | P0 P1 P2 P3"
    echo "  Phase 3 (loss)  : phase3 | L1 L2 L3 L4 L5 Lalt"
    echo "  Phase 4 (lr)    : phase4 | O1 O2 O3"
    echo "  Confirmation    : confirm | confirm_s44 confirm_s42 confirm_s123"
    echo ""
    echo "  Avant phases 2+ : éditer H_BEST / P_BEST / L_BEST / O_BEST"
    echo "                    en haut de ce script."
    echo ""
    ;;

  *)
    echo "Cible inconnue : '$TARGET'"
    echo "Lancer sans argument pour voir la liste."
    exit 1
    ;;
esac

