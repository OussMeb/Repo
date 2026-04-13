## 🔍 Analyse des Patients Exclus - Fichier ptv_channels.json

### 📊 Statistiques d'Exclusion

```
Total patients:     668
Patients exclus:     58  (8.7%)

Raisons:
├─ missing_ptv_json: 57 patients (98.3% des exclusions)
└─ missing_x:         1 patient  (1.7% des exclusions)
```

---

## 🗂️ D'où vient le fichier `ptv_channels.json` ?

### Origine: Pipeline de Préprocessing

Le fichier `ptv_channels.json` est **généré automatiquement** par votre pipeline de preprocessing (`preprocessing/pipeline.py`).

**Localisation dans le code**:
```python
# preprocessing/pipeline.py, ligne ~1758
ptv_meta = build_ptv_channel_metadata(
    ptv_clusters=ptv_clusters,
    vmin_gy=float(getattr(self, "ptv_norm_min_gy", PTV_NORM_MIN_GY)),
    vmax_gy=float(getattr(self, "ptv_norm_max_gy", PTV_NORM_MAX_GY)),
)
with open(patient_out_dir / "ptv_channels.json", "w", encoding="utf-8") as f:
    json.dump(_sanitize_for_json(ptv_meta), f, ensure_ascii=False, indent=2)
```

---

## 📝 Contenu du fichier ptv_channels.json

Ce fichier contient les métadonnées des PTVs (Planning Target Volumes) pour chaque patient:

```json
{
  "channels": {
    "ptv_low": {
      "doses_gy": [45.0, 50.4],
      "dose_rep_gy": 50.4,
      "dose_norm_rep": 0.05384615384615385,
      "n_files": 2,
      "files": ["PTV_45.nii.gz", "PTV_50_4.nii.gz"]
    },
    "ptv_mid": {
      "doses_gy": [60.0],
      "dose_rep_gy": 60.0,
      "dose_norm_rep": 0.42307692307692307,
      "n_files": 1,
      "files": ["PTV_60.nii.gz"]
    },
    "ptv_high": {
      "doses_gy": [70.0],
      "dose_rep_gy": 70.0,
      "dose_norm_rep": 0.8076923076923077,
      "n_files": 1,
      "files": ["PTV_70.nii.gz"]
    }
  },
  "clusters": [...]
}
```

---

## ❌ Pourquoi 57 Patients Sont Exclus ?

### Raison Principale: Fichier `ptv_channels.json` Manquant

Le fichier n'est **PAS créé** si le preprocessing échoue à l'étape des PTVs.

### Causes Possibles d'Échec:

#### 1. **Aucun PTV Valide Trouvé** (ligne 1688-1693)
```python
pts = find_ptv_candidates(struct_dir, preferred_structure_set_id=preferred_set_id)
if not pts:
    if preferred_set_id:
        logging.error(f"[PTV] No valid PTVs found in preferred structure set {preferred_set_id}")
        return _quarantine(f"no_ptv_in_structure_set_{preferred_set_id}")
    else:
        return _quarantine("aucun PTV valide")
```

**Signification**: 
- Le patient n'a **aucun fichier** de contour PTV dans ses structures
- OU les fichiers PTVs ne respectent pas le format de nommage attendu
- OU le `structure_set_id` préféré ne contient pas de PTVs

#### 2. **Clustering Impossible** (ligne 1698-1699)
```python
ptv_clusters = cluster_ptvs_by_dose(
    pts, tol_gy=float(getattr(self, "ptv_cluster_tol_gy", PTV_CLUSTER_TOL_GY))
)
if not ptv_clusters:
    return _quarantine("PTV: clustering impossible")
```

**Signification**:
- Les PTVs existent MAIS leur dose ne peut pas être extraite
- Problème de parsing du nom de fichier (ex: format non standard)
- Doses invalides ou aberrantes

#### 3. **PTV Mapping Requis mais Absent** (ligne 1680-1681)
```python
if self.require_ptv_mapping and not is_multi_group:
    return _quarantine("ptv_mapping_required_but_missing_or_invalid")
```

**Signification**:
- Votre config exige un mapping manuel des PTVs
- Le fichier de mapping est absent ou invalide

#### 4. **Exception lors de l'Écriture** (ligne 1760-1761)
```python
try:
    # ... écriture du fichier ...
except Exception as e:
    report["warnings"].append(f"ptv_channels.json write failed: {e}")
```

**Signification**:
- Le clustering a réussi MAIS l'écriture du fichier a échoué
- Problèmes de permissions, disque plein, etc.

---

## 🔍 Comment Diagnostiquer Vos 57 Patients ?

### Méthode 1: Vérifier les Logs de Preprocessing

Cherchez les messages d'erreur pour ces patients:

```bash
# Si vous avez des logs de preprocessing
grep -E "SKIP|quarantine|PTV.*No valid" preprocessing/logs/*.log

# Ou chercher les raisons d'exclusion
grep -E "aucun PTV valide|clustering impossible|ptv_mapping_required" preprocessing/logs/*.log
```

### Méthode 2: Vérifier Manuellement un Patient Exclu

```bash
# Exemple avec le patient 129363_group1
PATIENT_DIR="/mnt/LeGrosDisque/Julien/sianogramme/JUJU/129363_group1"

# 1. Vérifier si le fichier ptv_channels.json existe
ls -lh "$PATIENT_DIR/ptv_channels.json"  
# Si absent → preprocessing a échoué

# 2. Vérifier les structures disponibles
ls -lh "$PATIENT_DIR/structures/"*.nii.gz | grep -i ptv

# 3. Vérifier le report.json
cat "$PATIENT_DIR/report.json" | jq '.ptv'
# Ou chercher les warnings
cat "$PATIENT_DIR/report.json" | jq '.warnings'
```

### Méthode 3: Lister Tous les Patients Sans ptv_channels.json

```bash
cd /mnt/LeGrosDisque/Julien/sianogramme/JUJU

# Trouver tous les dossiers SANS ptv_channels.json
for d in */; do
    if [ ! -f "${d}ptv_channels.json" ] && [ ! -f "${d}structures/ptv_channels.json" ]; then
        echo "MISSING: $d"
        # Vérifier combien de PTVs sont présents
        count=$(find "${d}structures/" -name "*[Pp][Tt][Vv]*.nii.gz" 2>/dev/null | wc -l)
        echo "  → PTVs trouvés: $count"
    fi
done
```

---

## 💡 Solutions Possibles

### Solution 1: Réexécuter le Preprocessing pour les Patients Manquants

Si le preprocessing a échoué pour des raisons temporaires (permissions, etc.):

```bash
# Re-run preprocessing uniquement sur les patients manquants
python preprocessing/pipeline.py \
    --patients 129363_group1,284043_group0,... \
    --force
```

### Solution 2: Créer Manuellement les ptv_channels.json

Si les PTVs existent mais le fichier n'a pas été généré:

```python
from preprocessing.utils.ptv_utils import (
    find_ptv_candidates, 
    cluster_ptvs_by_dose,
    build_ptv_channel_metadata
)
import json
from pathlib import Path

patient_dir = Path("/mnt/LeGrosDisque/Julien/sianogramme/JUJU/129363_group1")
struct_dir = patient_dir / "structures"

# Trouver les PTVs
pts = find_ptv_candidates(struct_dir)
print(f"PTVs trouvés: {len(pts)}")

# Clustering
ptv_clusters = cluster_ptvs_by_dose(pts, tol_gy=2.0)
print(f"Clusters: {len(ptv_clusters)}")

# Générer metadata
ptv_meta = build_ptv_channel_metadata(
    ptv_clusters=ptv_clusters,
    vmin_gy=49.0,
    vmax_gy=75.0
)

# Sauvegarder
with open(patient_dir / "ptv_channels.json", "w") as f:
    json.dump(ptv_meta, f, indent=2)
```

### Solution 3: Accepter le Taux d'Exclusion

**Si 91.3% de vos patients sont chargés, c'est déjà excellent!**

Les 8.7% exclus peuvent être:
- Des cas cliniques particuliers sans PTVs standards
- Des données de test/calibration
- Des acquisitions incomplètes

---

## 📋 Checklist de Vérification

Pour chaque patient exclu, vérifiez:

- [ ] Le dossier `structures/` contient-il des fichiers `*PTV*.nii.gz` ?
- [ ] Les noms de fichiers PTVs sont-ils standards (ex: `PTV_70.nii.gz`) ?
- [ ] Le fichier `report.json` existe-t-il ?
- [ ] Le `report.json` contient-il des warnings sur les PTVs ?
- [ ] Le preprocessing a-t-il été exécuté jusqu'au bout ?
- [ ] Y a-t-il des erreurs de permissions sur le dossier ?

---

## 🎯 Format de Nommage Attendu pour les PTVs

Le code attend des noms comme:

```
✅ VALIDES:
PTV_70.nii.gz          → 70 Gy
PTV_60.nii.gz          → 60 Gy
CTV_54_Gy.nii.gz       → 54 Gy
PTV_52_8.nii.gz        → 52.8 Gy (underscore = point décimal)
PTV_69,96Gy.nii.gz     → 69.96 Gy (virgule française)

❌ NON RECONNUS:
structure_1.nii.gz     → Pas de dose extraite
ptv.nii.gz             → Pas de dose
PTV_high.nii.gz        → Pas de valeur numérique
```

---

## 🔧 Configuration Preprocessing à Vérifier

Dans votre config de preprocessing, vérifiez:

```python
# Est-ce que require_ptv_mapping est activé ?
require_ptv_mapping = False  # Si True, besoin d'un mapping manuel

# Tolérance de clustering
ptv_cluster_tol_gy = 2.0  # Gy (plus c'est grand, plus c'est permissif)

# Range de normalisation
ptv_norm_min_gy = 49.0
ptv_norm_max_gy = 75.0
```

---

**Conclusion**: Les 57 patients sont exclus car le fichier `ptv_channels.json` n'a pas été créé lors du preprocessing, très probablement à cause de l'**absence de PTVs valides** dans leurs structures. C'est un comportement normal du pipeline pour filtrer les données incomplètes ou non conformes.

