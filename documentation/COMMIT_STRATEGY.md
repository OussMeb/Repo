# Stratégie Git - Simplification Model

## 📍 État actuel
- **Branche:** `simplification_model`
- **Fichiers principaux:** `model_simplified.py`, `dataloader_patches_simplified.py`
- **But final:** Écraser `main` après validation complète

## 🎯 Plan recommandé

### Phase 1 : Commit sur simplification_model ✅ (MAINTENANT)
```bash
# 1. Ajouter tous les nouveaux fichiers
git add model_simplified.py dataloader_patches_simplified.py
git add quick_test_juju.py launch_training_juju.py
git add regenerate_ptv_channels.py validate_juju_dataset.py
git add check_training_progress.py demo_rgb_viz.py
git add *.md  # Toute la documentation

# 2. Ajouter les corrections de bugs
git add preprocessing/utils/ptv_utils.py

# 3. Commit avec message clair
git commit -m "feat: Complete model simplification with modern training approach

Major refactoring:
- NEW: model_simplified.py (modern training loop, no build_default)
- NEW: RGB visualization (3 DRR channels instead of 16)
- NEW: Proper data loading with patches
- FIX: ptv_channels.json bug (+57 patients recovered)
- DOC: Complete guides and documentation

Stats:
- 667/668 patients usable (99.85%)
- 4.99M parameters
- Modern PyTorch patterns (amp, scheduler, etc.)

Related files:
- model_simplified.py
- quick_test_juju.py
- launch_training_juju.py
- regenerate_ptv_channels.py
- validate_juju_dataset.py
- preprocessing/utils/ptv_utils.py (bug fix)
- Documentation: GUIDE_*.md, BUG_FIX_*.md
"

# 4. Push
git push origin simplification_model
```

### Phase 2 : Tests et validation (1-2 semaines)
- Entraîner plusieurs modèles
- Comparer performances
- Valider stabilité
- Documenter résultats

### Phase 3 : Renommage _simplified → base (après validation)
```bash
# Renommer les fichiers principaux
git mv model_simplified.py model.py  # Écrase l'ancien
git mv dataloader_patches_simplified.py dataloader_patches.py  # Si existe

git commit -m "refactor: Replace old implementation with simplified version

After complete validation, replacing old model.py with simplified version.
Old version preserved in git history.
"
```

### Phase 4 : Merge dans main (après validation complète)
```bash
# S'assurer que tout est commité
git status

# Basculer sur main
git checkout main

# Merger simplification_model
git merge simplification_model

# Résoudre conflits si nécessaire

# Push
git push origin main
```

## 📦 Fichiers à commiter maintenant

### Code principal
- ✅ `model_simplified.py` (4.99M params, modern training)
- ✅ `quick_test_juju.py` (test rapide 5 epochs)
- ✅ `launch_training_juju.py` (entraînement complet)

### Utilitaires
- ✅ `regenerate_ptv_channels.py` (fix bug dataset)
- ✅ `validate_juju_dataset.py` (validation dataset)
- ✅ `check_training_progress.py` (monitoring)
- ✅ `demo_rgb_viz.py` (démo visualisation RGB)

### Bug fixes
- ✅ `preprocessing/utils/ptv_utils.py` (fix ptv_channels.json)

### Documentation
- ✅ `GUIDE_JUJU.md` (guide complet)
- ✅ `GUIDE_REFERENCE_RAPIDE.md` (référence rapide)
- ✅ `BUG_FIX_ptv_channels.md` (doc bug fix)
- ✅ `RESOLUTION_COMPLETE.md` (résumé bug fix)
- ✅ `MODIFICATIONS_VISUALS.md` (doc visualisation)
- ✅ `SUMMARY_RGB_VISUALIZATION.md` (résumé RGB viz)
- ✅ `ANALYSE_CHARGEMENT_DATASET.md` (analyse dataset)
- ✅ `ANALYSE_EXCLUSIONS_PATIENTS.md` (analyse exclusions)
- ✅ `REFACTORING_SUMMARY.md` (résumé refactoring)

### À NE PAS commiter
- ❌ `*.pyc`, `__pycache__/` (déjà dans .gitignore normalement)
- ❌ `preprocessing/test_run.log` (fichier temporaire)
- ❌ `model_backup.py` (backup local uniquement)

## 🎯 Commandes recommandées

```bash
# Nettoyer les fichiers pycache
git restore preprocessing/utils/__pycache__/*.pyc

# Ajouter tous les fichiers pertinents
git add model_simplified.py
git add quick_test_juju.py launch_training_juju.py
git add regenerate_ptv_channels.py validate_juju_dataset.py
git add check_training_progress.py demo_rgb_viz.py
git add preprocessing/utils/ptv_utils.py
git add GUIDE_*.md BUG_FIX_*.md RESOLUTION_COMPLETE.md
git add MODIFICATIONS_VISUALS.md SUMMARY_RGB_VISUALIZATION.md
git add ANALYSE_*.md REFACTORING_SUMMARY.md

# Vérifier ce qui sera commité
git status

# Commit
git commit -m "feat: Complete model simplification and bug fixes

Major improvements:
- Complete model refactoring (model_simplified.py)
- RGB visualization (3 channels instead of 16)
- Modern training patterns (AMP, scheduler, EMA option)
- Bug fix: ptv_channels.json (+57 patients)
- Complete documentation and guides

Dataset improvements:
- 667/668 patients usable (99.85%)
- +9.3% training data recovered

Model improvements:
- 4.99M parameters
- Cleaner architecture
- Better visualization
- Easier to maintain and debug
"

# Push
git push origin simplification_model
```

## 📊 Pourquoi garder _simplified maintenant ?

### Avantages ✅
1. **Traçabilité** : Comparaison facile ancien vs nouveau
2. **Sécurité** : Pas de risque de casser main prématurément
3. **Flexibilité** : Possibilité de tester les deux approches
4. **Historique clair** : Git log montre l'évolution

### Quand renommer ?
- ✅ Après 5-10 entraînements complets
- ✅ Après validation des performances
- ✅ Après validation de la stabilité
- ✅ Quand l'équipe est d'accord

### Comment renommer ?
```bash
# Dans la branche simplification_model
git mv model_simplified.py model.py
git commit -m "refactor: Finalize simplified model as main implementation"
git push origin simplification_model

# Puis merger dans main quand prêt
```

## ⚠️ Points d'attention

### Avant de merger dans main
- [ ] Vérifier que tous les tests passent
- [ ] Documenter les changements breaking
- [ ] Prévenir l'équipe
- [ ] Faire un backup de main
- [ ] Tester le merge en local d'abord

### Fichiers à garder en local (pas commit)
- `model_backup.py`
- `preprocessing/test_run.log`
- Tous les `__pycache__/`

## 🎓 Résumé

**MAINTENANT:** 
- Commiter sur `simplification_model` AVEC le suffixe `_simplified`
- Garder les anciens fichiers intacts dans `main`

**PLUS TARD (après validation):**
- Renommer `_simplified` → fichiers de base
- Merger dans `main`
- Archiver les anciennes versions

**AVANTAGE:** Sécurité maximale + traçabilité complète

