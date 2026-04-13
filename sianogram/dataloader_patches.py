# dataloader_patches_v2026.py
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Kornia (optionnel : si non dispo, on désactive les aug géométriques kornia)
try:
    import kornia.augmentation as K
except Exception:
    K = None

LOGGER = logging.getLogger(__name__)

# Batch contract unique train/val/test.
BATCH_KEYS = (
    "x_drr",
    "angles",
    "positions",
    "y_sino",
    "film",
    "ptv_dose_norm",
    "ptv_present",
    "cp_dur_sec_mean",
    "patient_number",
    "core_start_cp",
    "core_end_cp",
)


# =========================
# utils
# =========================


def _read_json_strict(path: Path, who: str):
    try:
        with path.open("r") as f:
            j = json.load(f)
    except Exception as e:
        raise RuntimeError(f"[SKIP:{who}] JSON illisible: {path} ({e})")
    if not isinstance(j, dict):
        raise RuntimeError(f"[SKIP:{who}] JSON non-dict: {path}")
    return j

def _extract_cp_duration_strict(plan_summary_file: Path, who: str) -> float:
    j = _read_json_strict(plan_summary_file, who)
    plan = j.get("plan", None)
    if not isinstance(plan, dict):
        raise RuntimeError(f"[SKIP:{who}] plan_summary sans 'plan' dict: {plan_summary_file}")
    v = plan.get("cp_duration_sec_mean", None)
    if v is None:
        raise RuntimeError(f"[SKIP:{who}] plan_summary sans cp_duration_sec_mean: {plan_summary_file}")
    try:
        v = float(v)
    except Exception:
        raise RuntimeError(f"[SKIP:{who}] cp_duration_sec_mean non numérique: {plan_summary_file} ({v})")
    if not np.isfinite(v):
        raise RuntimeError(f"[SKIP:{who}] cp_duration_sec_mean non fini: {plan_summary_file} ({v})")
    return v

def _norm01_strict(v: float, vmin: float, vmax: float, who: str, label: str) -> float:
    if vmin is None or vmax is None:
        raise RuntimeError(f"[CONFIG] {label} norm: vmin/vmax doivent être fournis (pas de défaut).")
    vmin = float(vmin); vmax = float(vmax)
    if vmax <= vmin:
        raise RuntimeError(f"[CONFIG] {label} norm: vmax<=vmin ({vmax} <= {vmin})")
    x = (float(v) - vmin) / (vmax - vmin)
    if not np.isfinite(x):
        raise RuntimeError(f"[SKIP:{who}] {label} norm non finie: v={v} vmin={vmin} vmax={vmax}")
    if x < 0.0 or x > 1.0:
        LOGGER.warning(
            "[WARN:%s] %s hors [0,1] apres norm -> clip. v=%.6g -> %.3f (vmin=%.6g, vmax=%.6g)",
            who,
            label,
            float(v),
            float(x),
            vmin,
            vmax,
        )
    return float(np.clip(x, 0.0, 1.0))

def _norm01_margin(v: float, vmin: float, vmax: float, who: str = None, label: str = None, margin: float = 0.2) -> float:
    """
    Normalise avec margin hors plage, puis remap vers [0,1].

    Permet aux valeurs hors [vmin, vmax] de sortir de [0,1], mais avec une bande de tolérance.
    Logging optionnel si valeur hors plage détectée.

    Args:
        v: valeur à normaliser
        vmin: minimum de la plage nominale
        vmax: maximum de la plage nominale
        who: identifiant pour logging (optionnel)
        label: label pour logging (optionnel)
        margin: marge de tolérance (défaut 0.2 = 20%)

    Returns:
        Valeur normalisée et remappée dans [0, 1]
    """
    if vmin is None or vmax is None:
        raise RuntimeError(f"[CONFIG] {label or 'value'} norm: vmin/vmax doivent être fournis.")
    vmin = float(vmin); vmax = float(vmax)
    if vmax <= vmin:
        raise RuntimeError(f"[CONFIG] {label or 'value'} norm: vmax<=vmin ({vmax} <= {vmin})")

    v = float(v)
    if not np.isfinite(v):
        raise RuntimeError(f"[SKIP:{who}] {label or 'value'} norm non finie: v={v}")

    # Normalisation brute (peut sortir de [0,1])
    x = (v - vmin) / (vmax - vmin)


    # Appliquer la margin: restreindre à [-margin, 1+margin]
    if x < -margin:
        x = -margin
    elif x > 1.0 + margin:
        x = 1.0 + margin

    # Remap vers [0,1] en conservant l'ordre
    # Si x ∈ [-margin, 1+margin], alors (x + margin) ∈ [0, 1+2*margin]
    # Donc (x + margin) / (1 + 2*margin) ∈ [0, 1]
    x = (x + margin) / (1.0 + 2.0 * margin)

    # Clamp final de sécurité (ne devrait pas être nécessaire)
    x = float(np.clip(x, 0.0, 1.0))
    return x

def _extract_ptv_doses_with_messages(
    ptv_channels_file: Path,
    who: str,
    dose_min_gy: float,
    dose_max_gy: float,
):
    """
    Supporte 2 formats :
      A) {"ptv_br": {...}, "ptv_ri": {...}, "ptv_hr": {...}}
      B) {"channels": {"ptv_br": {...}, ...}, "clusters": [...]}

    Règles (STRICTES):
      - PTV absent => present=0, dose_norm=0 (ok)
      - PTV présent (dict existe) mais dose manquante/non-numérique => SKIP (incohérent!)
      - PTV présent et dose valide => present=1, dose_norm normalisé avec margin
      - Check final: si present==0 => dose_norm doit être 0
    """
    if dose_min_gy is None or dose_max_gy is None:
        raise RuntimeError("[CONFIG] dose_min_gy/dose_max_gy doivent être fournis (pas de défaut).")
    dose_min_gy = float(dose_min_gy)
    dose_max_gy = float(dose_max_gy)
    if dose_max_gy <= dose_min_gy:
        raise RuntimeError(f"[CONFIG] dose_max_gy<=dose_min_gy ({dose_max_gy} <= {dose_min_gy})")

    j = _read_json_strict(ptv_channels_file, who)

    ch = j.get("channels", None)
    if ch is None:
        channels = j  # ancien format (ptv_* au root)
    else:
        if not isinstance(ch, dict):
            raise RuntimeError(f"[SKIP:{who}] 'channels' présent mais pas dict: {ptv_channels_file}")
        channels = ch

    names = ["ptv_br", "ptv_ri", "ptv_hr"]
    doses = np.zeros((3,), dtype=np.float32)
    present = np.zeros((3,), dtype=np.float32)

    for i, name in enumerate(names):
        d = channels.get(name, None)

        if not isinstance(d, dict):
            # PTV absent
            # LOGGER.info("[INFO:%s] PTV %s absent -> dose=0, present=0", who, name)  # Supprimé: trop verbeux
            present[i] = 0.0
            doses[i] = 0.0
            continue

        # PTV présent (dict existe)
        present[i] = 1.0
        gy = d.get("dose_rep_gy", None)

        # ===== RÈGLE STRICTE: présent + dose manquante = SKIP (incohérent) =====
        if gy is None:
            raise RuntimeError(
                f"[SKIP:{who}] PTV {name} présent (dict existe) mais dose_rep_gy manquante -> INCOHÉRENT"
            )

        try:
            gy = float(gy)
        except Exception as e:
            raise RuntimeError(
                f"[SKIP:{who}] PTV {name} présent mais dose_rep_gy non numérique ({gy}) -> INCOHÉRENT"
            )

        if not np.isfinite(gy):
            raise RuntimeError(
                f"[SKIP:{who}] PTV {name} présent mais dose_rep_gy non finie ({gy}) -> INCOHÉRENT"
            )

        doses[i] = gy

    # ===== Normaliser les doses avec margin =====
    doses_norm = np.zeros((3,), dtype=np.float32)
    for i in range(3):
        # PTV absent: garder 0 sans passer par la normalisation (évite logs outlier inutiles)
        if present[i] == 0.0:
            doses_norm[i] = 0.0
            continue
        doses_norm[i] = _norm01_margin(
            doses[i],
            dose_min_gy,
            dose_max_gy,
            who=who,
            label=f"ptv_{names[i]}_dose_gy",
            margin=0.2
        )

    # ===== Check de cohérence final: present <-> dose_norm =====
    for i in range(3):
        if present[i] == 0.0 and doses_norm[i] != 0.0:
            # PTV absent mais dose calculée => forcer dose à 0
            LOGGER.warning(
                "[WARN:%s] Cohérence: PTV %s présent=0 mais dose_norm=%.3f -> force à 0",
                who, names[i], doses_norm[i]
            )
            doses_norm[i] = 0.0
        elif present[i] == 1.0 and doses_norm[i] == 0.0:
            # PTV présent mais dose_norm=0 (suspect mais toléré après normalisation)
            LOGGER.info(
                "[INFO:%s] Cohérence: PTV %s présent=1 mais dose_norm=0 (after norm) -> ok",
                who, names[i]
            )

    return doses_norm, present


def _find_existing(*candidates: Path):
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


# Legacy helpers removed: _read_json_cached, _load_ptv_dose_norm_from_ptv_channels,
# _load_cp_duration_sec_mean, _load_aux_plan_info.
# _collect_subjects is now the unique source of truth for aux plan/PTV extraction.


def _collect_subjects(
    path_root: str,
    allowed_ids: set[str] | None = None,
    *,
    dose_min_gy: float,
    dose_max_gy: float,
    cp_dur_min_sec: float,
    cp_dur_max_sec: float,
    film_with_presence: bool = True,
):
    """
    Scan dataset et construit subjects (pré-calcul film/ptv/cp_dur) en STRICT.

    Requis (sinon SKIP):
      - X_montage.npy
      - sino.npy
      - ptv_channels.json (lisible)
      - plan_summary.json avec plan.cp_duration_sec_mean (lisible)
      - cp_dur_min_sec/cp_dur_max_sec fournis (pas de défaut)
      - dose_min_gy/dose_max_gy fournis (pas de défaut)

    Règles de cohérence (SKIP si violation):
      - PTV absent => present=0, dose_norm=0 (OK)
      - PTV présent (dict existe) mais dose manquante/non-numérique => SKIP (INCOHÉRENT!)

    Normalisation:
      - Doses et cp_dur normalisés avec margin=0.2 (permet outliers, pas de hard clip)
      - Valeurs hors plage nominale sont remappées vers [0,1] en conservant l'ordre
      - Check final: si present==0 => dose_norm doit rester 0
    """
    # ---- garde-fous config (zéro défaut) ----
    for k, v in [
        ("dose_min_gy", dose_min_gy),
        ("dose_max_gy", dose_max_gy),
        ("cp_dur_min_sec", cp_dur_min_sec),
        ("cp_dur_max_sec", cp_dur_max_sec),
    ]:
        if v is None:
            raise RuntimeError(f"[CONFIG] {k} doit être fourni (pas de valeur par défaut).")

    dose_min_gy = float(dose_min_gy)
    dose_max_gy = float(dose_max_gy)
    if dose_max_gy <= dose_min_gy:
        raise RuntimeError(f"[CONFIG] dose_max_gy<=dose_min_gy ({dose_max_gy} <= {dose_min_gy})")

    cp_dur_min_sec = float(cp_dur_min_sec)
    cp_dur_max_sec = float(cp_dur_max_sec)
    if cp_dur_max_sec <= cp_dur_min_sec:
        raise RuntimeError(f"[CONFIG] cp_dur_max_sec<=cp_dur_min_sec ({cp_dur_max_sec} <= {cp_dur_min_sec})")

    subs = []
    root = Path(path_root)

    skip_counts = {}
    cp_vals = []
    n_warn_clip_cp = 0

    def _skip(who: str, msg: str, reason: str):
        LOGGER.warning("[SKIP:%s] %s", who, msg)
        skip_counts[reason] = skip_counts.get(reason, 0) + 1

    def sort_key(name: str):
        return (not name.isdigit(), int(name) if name.isdigit() else name)

    for name in sorted([d.name for d in root.iterdir() if d.is_dir()], key=sort_key):
        if allowed_ids is not None and name not in allowed_ids:
            continue

        who = str(name)
        d = root / name

        fx = d / "X_montage.npy"
        fy = d / "sino.npy"
        fa = d / "angles_real_deg.npy"

        if not fx.is_file():
            _skip(who, f"X manquant: {fx}", "missing_x")
            continue
        if not fy.is_file():
            _skip(who, f"Y manquant: {fy}", "missing_y")
            continue

        structures_dir = d / "structures"
        if not structures_dir.is_dir():
            structures_dir = d

        ptv_json = _find_existing(structures_dir / "ptv_channels.json", d / "ptv_channels.json")
        if ptv_json is None:
            _skip(who, "ptv_channels.json introuvable", "missing_ptv_json")
            continue

        plan_json = _find_existing(d / "plan_summary.json", d / "plan_summary" / "plan_summary.json")
        if plan_json is None:
            _skip(who, "plan_summary.json introuvable", "missing_plan_json")
            continue

        # --------- lecture stricte plan_summary -> cp_dur ---------
        try:
            cp_dur = _extract_cp_duration_strict(Path(plan_json), who=who)
        except Exception as e:
            _skip(who, str(e), "bad_plan_json")
            continue

        # --------- lecture stricte ptv_channels + doses tolérées ---------
        try:
            doses_norm, present = _extract_ptv_doses_with_messages(
                Path(ptv_json),
                who=who,
                dose_min_gy=dose_min_gy,
                dose_max_gy=dose_max_gy,
            )
        except Exception as e:
            # fichier illisible / structure invalide => on exclut
            _skip(who, str(e), "bad_ptv_json")
            continue

        # --------- normalisation cp_dur avec margin (support outliers) ---------
        try:
            cp_dur_norm = _norm01_margin(
                float(cp_dur),
                cp_dur_min_sec,
                cp_dur_max_sec,
                who=who,
                label="cp_dur_sec_mean",
                margin=0.2
            )
        except Exception as e:
            _skip(who, str(e), "bad_cp_norm")
            continue

        cp_vals.append(float(cp_dur))

        # --------- film (4 ou 7) ---------
        if film_with_presence:
            film = np.concatenate(
                [doses_norm.astype(np.float32), np.array([cp_dur_norm], np.float32), present.astype(np.float32)],
                axis=0
            ).astype(np.float32)  # (7,)
        else:
            film = np.concatenate(
                [doses_norm.astype(np.float32), np.array([cp_dur_norm], np.float32)],
                axis=0
            ).astype(np.float32)  # (4,)

        subs.append(
            {
                "x_file": str(fx),
                "y_file": str(fy),
                "gantry_file": str(fa) if fa.is_file() else None,
                "patient_number": name,

                "ptv_channels_file": str(ptv_json),
                "plan_summary_file": str(plan_json),

                # pré-calculés (0 IO en __getitem__)
                "film": film.tolist(),
                "ptv_dose_norm": doses_norm.astype(np.float32).tolist(),
                "ptv_present": present.astype(np.float32).tolist(),
                "cp_dur_sec_mean": float(cp_dur),
            }
        )

    # --------- résumé ---------
    LOGGER.info("[DATA] sujets conserves: %d", len(subs))
    if skip_counts:
        LOGGER.info("[DATA] sujets exclus (raison -> count):")
        for k in sorted(skip_counts.keys()):
            LOGGER.info("  - %s: %d", k, skip_counts[k])

    if cp_vals:
        cp_vals_np = np.asarray(cp_vals, dtype=np.float32)
        LOGGER.info(
            "[DATA] cp_dur_sec_mean stats: min=%.6g max=%.6g p5=%.6g p95=%.6g (clipped_norm_count=%d)",
            float(cp_vals_np.min()),
            float(cp_vals_np.max()),
            float(np.percentile(cp_vals_np, 5)),
            float(np.percentile(cp_vals_np, 95)),
            int(n_warn_clip_cp),
        )

    return subs


def _load_split(split_json: str | None, subset: str | None) -> set[str] | None:
    """
    Charge le manifeste JSON s’il est fourni.
    subset ∈ {'train','val','test'} ou None.
    Retourne un set d’IDs patients à conserver, ou None si pas de split fourni.
    """
    if not split_json:
        return None
    with open(split_json, "r") as f:
        split = json.load(f)
    if subset is None:
        return None
    ids = split.get(subset, [])
    return set(str(x) for x in ids)


def _center_resize_width(t: torch.Tensor, new_W: int) -> torch.Tensor:
    """
    Redimensionne uniquement en largeur vers new_W, puis recadre/pad au centre
    pour revenir à la largeur d'origine. t: [C,H,W] (inclut le cas [1,H,W]).
    """
    assert t.ndim == 3, f"_center_resize_width attend [C,H,W], reçu {tuple(t.shape)}"
    C, H, W = t.shape
    if new_W == W:
        return t

    t_scaled = F.interpolate(
        t.unsqueeze(0),
        size=(H, new_W),
        mode="bilinear",
        align_corners=False,
    )[0]

    if new_W > W:
        start = max(0, (new_W - W) // 2)
        return t_scaled[..., start : start + W]

    pad_left = (W - new_W) // 2
    pad_right = W - new_W - pad_left
    return F.pad(t_scaled, (pad_left, pad_right, 0, 0), value=0.0)


def _to_tensor_f32(a: np.ndarray) -> torch.Tensor:
    """
    Convertit un np.ndarray (y compris memmap/readonly) en Tensor float32
    avec un buffer propre (copie garantie).
    """
    a_c = np.array(a, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(a_c)


def _load_gantry_norm(
    gantry_file: str | None,
    n_cp_needed: int,
    start_cp: int,
    fallback_delta_deg: float = 360.0 / 51.0,
) -> torch.Tensor:
    """
    Lit un vecteur d'angles en degrés aligné CP, le découpe sur [start_cp : start_cp + n_cp_needed],
    puis le normalise en [0,1) via division par 360.

    Supporte start_cp négatif :
      CP < 0 paddés avec la valeur CP=0 (première valeur existante)
      CP > fin paddés avec la dernière valeur

    Si gantry_file est absent/invalide, génère une rampe synthétique au pas fallback_delta_deg,
    avec indices clampés min=0 (donc padding cp<0 équivalent à cp=0).

    Retour: Tensor shape [n_cp_needed, 1]
    """
    if gantry_file is not None:
        try:
            arr = np.load(gantry_file, mmap_mode="r")
            arr = np.asarray(arr, dtype=np.float32).reshape(-1)

            if arr.size == 0:
                raise RuntimeError("gantry_angles_deg.npy vide")

            pad_front = max(0, -int(start_cp))
            start0 = max(0, int(start_cp))

            take = n_cp_needed - pad_front
            end0 = start0 + take

            if end0 <= arr.shape[0]:
                sl = arr[start0:end0]
                pad_end = 0
            else:
                sl = arr[start0:] if start0 < arr.shape[0] else np.empty((0,), dtype=np.float32)
                pad_end = max(0, end0 - arr.shape[0])

            if pad_front > 0:
                front = np.full((pad_front,), arr[0], dtype=np.float32)
            else:
                front = np.empty((0,), dtype=np.float32)

            if pad_end > 0:
                back = np.full((pad_end,), arr[-1], dtype=np.float32)
            else:
                back = np.empty((0,), dtype=np.float32)

            sl = np.concatenate([front, sl, back], axis=0)[:n_cp_needed]
            sl = (sl % 360.0) / 360.0
            return torch.from_numpy(sl).view(n_cp_needed, 1).contiguous()

        except Exception:
            pass

    cp_idx = torch.arange(start_cp, start_cp + n_cp_needed, dtype=torch.float32)
    cp_idx = torch.clamp(cp_idx, min=0.0)
    ang = ((cp_idx * fallback_delta_deg) % 360.0) / 360.0
    return ang.view(n_cp_needed, 1).contiguous()


def _build_batch_dict(
    info: dict,
    x_drr: torch.Tensor,
    angles: torch.Tensor,
    positions: torch.Tensor,
    y_sino: torch.Tensor,
    core_start_cp: int | None = None,
    core_end_cp: int | None = None,
) -> dict:
    """Construit un batch au format standard train/val/test (voir BATCH_KEYS)."""
    y_cp = int(y_sino.shape[-2])
    core_s = int(core_start_cp) if core_start_cp is not None else 0
    core_e = int(core_end_cp) if core_end_cp is not None else y_cp
    core_s = max(0, min(core_s, y_cp))
    core_e = max(core_s, min(core_e, y_cp))

    return {
        "x_drr": x_drr,
        "angles": angles,
        "positions": positions,
        "y_sino": y_sino,
        "film": torch.tensor(info["film"], dtype=torch.float32).contiguous(),
        "ptv_dose_norm": torch.tensor(info["ptv_dose_norm"], dtype=torch.float32).contiguous(),
        "ptv_present": torch.tensor(info["ptv_present"], dtype=torch.float32).contiguous(),
        "cp_dur_sec_mean": torch.tensor([float(info["cp_dur_sec_mean"])], dtype=torch.float32),
        "patient_number": info["patient_number"],
        "core_start_cp": torch.tensor(core_s, dtype=torch.long),
        "core_end_cp": torch.tensor(core_e, dtype=torch.long),
    }


# =========================
# Dataset (train, patches)
# =========================
class SinogramPatchAugmentedDataset(Dataset):
    """
    Patches sinogramme + DRR synchrones.
    """

    def __init__(
        self,
        subjects,
        W_out: int,
        W_in: int | None = None,
        cp_height_px: int = 8,
        cp_unit: int | None = None,
        patch_in_cp: int = 256,
        patch_out_cp: int | None = None,
        halo_cp: int = 0,
        jitter_cp: int = 32,
        augment: bool = True,
        mmap: bool = True,
    ):
        self.subjects = subjects

        # W_out = largeur du sinogramme (64)
        self.W = int(W_out)

        # W_in = largeur du DRR/montage (128). Si None, on déduit depuis X_montage.npy
        self.W_in = int(W_in) if W_in is not None else None

        if cp_unit is not None and int(cp_unit) != int(cp_height_px):
            raise ValueError(f"cp_unit ({cp_unit}) doit egaler cp_height_px ({cp_height_px})")
        self.cp_height_px = int(cp_height_px if cp_unit is None else cp_unit)
        self.cp_unit = self.cp_height_px  # alias legacy interne
        self.patch_in_cp = int(patch_in_cp)
        self.patch_out_cp = int(patch_out_cp) if patch_out_cp is not None else int(patch_in_cp)
        self.halo_cp = int(halo_cp)
        if self.patch_in_cp <= 0 or self.patch_out_cp <= 0:
            raise ValueError("patch_in_cp et patch_out_cp doivent etre > 0")
        if self.patch_in_cp < self.patch_out_cp:
            raise ValueError("patch_in_cp doit etre >= patch_out_cp")
        expected_in = self.patch_out_cp + 2 * self.halo_cp
        if expected_in != self.patch_in_cp:
            raise ValueError(
                f"patch_in_cp incoherent: attendu patch_out_cp + 2*halo_cp = {expected_in}, recu {self.patch_in_cp}"
            )
        self.jitter_cp = int(jitter_cp)
        self.augment = bool(augment)
        self.mmap = bool(mmap)

        # Kornia désactivé (car x et y n'ont plus la même largeur)
        self.transform = None

        # Translation horizontale (en fraction de la largeur physique, cohérente entre 128 et 64)
        self.trans_p = 0.15
        self.translate_frac = 0.05

        self.hflip_p = 0.03
        self.scale_p = 0.35
        self.scale_min = 0.80
        self.scale_max = 1.20

        self.elastic_p = 0.15
        self.elastic_ctrl = 8
        self.elastic_max_px = 1.5
        self.elastic_smooth = 9

        self.cdrop_p = 0.35
        self.cdrop_prob = 0.12
        self.cdrop_start = 3

        self._n_patches = []
        for info in tqdm(self.subjects, desc="Loading training patches"):
            drr = np.load(info["x_file"], mmap_mode="r") if self.mmap else np.load(info["x_file"])
            Lpx = int(drr.shape[1])
            n_cp = Lpx // self.cp_height_px
            n_p = 1 if n_cp <= 0 else math.ceil(max(1, n_cp) / self.patch_out_cp)
            self._n_patches.append(n_p)

    def __len__(self):
        return sum(self._n_patches)

    def _flip_angles_norm01(self, angles: torch.Tensor) -> torch.Tensor:
        one = torch.ones_like(angles)
        return torch.remainder(one - angles, 1.0)

    def _resample_1d(self, v: torch.Tensor, new_W: int) -> torch.Tensor:
        """
        v: [W] float tensor -> resample to [new_W] (linéaire)
        """
        if v.numel() == new_W:
            return v
        vv = v.view(1, 1, -1)
        vv = F.interpolate(vv, size=new_W, mode="linear", align_corners=False)
        return vv.view(-1)

    def _make_dx_norm_base(self, W_base: int, device, dtype) -> torch.Tensor:
        """
        Génère un dx horizontal en coordonnées normalisées [-1,1], longueur W_base.
        Amplitude contrôlée par elastic_max_px (en pixels) convertie en norme.
        """
        if W_base <= 2:
            return torch.zeros((W_base,), device=device, dtype=dtype)

        amp_px = random.uniform(0.0, float(self.elastic_max_px))
        if amp_px <= 0.0:
            return torch.zeros((W_base,), device=device, dtype=dtype)

        n_ctrl = max(2, int(self.elastic_ctrl))
        ctrl = torch.randn((1, 1, 1, n_ctrl), device=device, dtype=dtype)
        ctrl = ctrl / ctrl.abs().amax().clamp_min(1e-6)

        dx = F.interpolate(ctrl, size=(1, W_base), mode="bilinear", align_corners=False).view(W_base)
        dx = dx * amp_px

        k = int(getattr(self, "elastic_smooth", 0))
        if k and k >= 3 and k % 2 == 1:
            w = torch.ones((1, 1, k), device=device, dtype=dtype) / float(k)
            dx_ = dx.view(1, 1, W_base)
            dx = F.conv1d(F.pad(dx_, (k // 2, k // 2), mode="replicate"), w).view(W_base)

        dx_norm = dx * (2.0 / float(max(W_base - 1, 1)))
        return dx_norm

    def _warp_lr_only(self, t: torch.Tensor, scale: float, shift_norm: float,
                      dx_norm_base: torch.Tensor | None) -> torch.Tensor:
        """
        Warp horizontal uniquement via grid_sample.
        t: [C,H,W]
        scale: facteur horizontal (>0)
        shift_norm: translation en coordonnées normalisées [-1,1]
        dx_norm_base: vecteur dx en coordonnées normalisées, défini sur une base W_base (ou None)
        """
        C, H, W = t.shape
        if W <= 2:
            return t

        device = t.device
        dtype = t.dtype

        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        if scale <= 0.0:
            scale = 1.0

        grid_x = (grid_x - float(shift_norm)) / float(scale)

        if dx_norm_base is not None and dx_norm_base.numel() > 0:
            dx = self._resample_1d(dx_norm_base.to(device=device, dtype=dtype), W)
            grid_x = grid_x + dx.view(1, W)

        grid_x = grid_x.clamp(-1.0, 1.0)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        out = F.grid_sample(
            t.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return out[0]

    def _apply_pair_augs(
            self,
            x: torch.Tensor,
            y_full: torch.Tensor,
            angles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [C,H,W_in]
        y_full: [1,H,W_out]
        angles: [patch_cp,1]
        Applique les mêmes paramètres d'aug en "coordonnées normalisées" aux deux largeurs.
        """
        do_flip = (self.hflip_p > 0.0) and (random.random() < self.hflip_p)

        if do_flip:
            x = torch.flip(x, dims=[2])
            y_full = torch.flip(y_full, dims=[2])
            angles = self._flip_angles_norm01(angles)

        scale = 1.0
        if (self.scale_p > 0.0) and (random.random() < self.scale_p):
            scale = float(random.uniform(self.scale_min, self.scale_max))

        shift_norm = 0.0
        if (getattr(self, "trans_p", 0.0) > 0.0) and (random.random() < float(self.trans_p)):
            max_shift_norm = 2.0 * float(getattr(self, "translate_frac", 0.0))
            shift_norm = float(random.uniform(-max_shift_norm, max_shift_norm))

        dx_base = None
        if (self.elastic_p > 0.0) and (random.random() < self.elastic_p):
            W_base = int(max(x.shape[-1], y_full.shape[-1]))
            dx_base = self._make_dx_norm_base(W_base=W_base, device=x.device, dtype=x.dtype)

        need_warp = (scale != 1.0) or (shift_norm != 0.0) or (dx_base is not None)
        if need_warp:
            x = self._warp_lr_only(x, scale=scale, shift_norm=shift_norm, dx_norm_base=dx_base)
            y_full = self._warp_lr_only(y_full, scale=scale, shift_norm=shift_norm, dx_norm_base=dx_base)

        return x, y_full, angles


    def _apply_channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.cdrop_p <= 0.0 or random.random() >= self.cdrop_p:
            return x

        C = x.shape[0]
        c0 = int(self.cdrop_start)
        if C <= c0:
            return x

        n = C - c0
        keep = (torch.rand(n, device=x.device) >= float(self.cdrop_prob)).to(dtype=x.dtype).view(n, 1, 1)

        mask = torch.ones((C, 1, 1), device=x.device, dtype=x.dtype)
        mask[c0:] = keep
        return x * mask

    def __getitem__(self, idx):
        cum = 0
        for pid, n_p in enumerate(self._n_patches):
            if idx < cum + n_p:
                patch_id = idx - cum
                break
            cum += n_p
        info = self.subjects[pid]

        # ---- load X/Y (comme avant) ----
        drr = np.load(info["x_file"], mmap_mode="r") if self.mmap else np.load(info["x_file"])
        sino = np.load(info["y_file"], mmap_mode="r") if self.mmap else np.load(info["y_file"])
        drr = drr.astype(np.float32, copy=False)
        sino = sino.astype(np.float32, copy=False)

        C, Lpx, Wp = drr.shape
        n_cp = Lpx // self.cp_height_px

        if sino.ndim == 2 and sino.shape[0] == n_cp and Lpx == n_cp * self.cp_unit:
            sino = np.repeat(sino, self.cp_unit, axis=0)

        window_px = self.patch_in_cp * self.cp_height_px

        n_patches = self._n_patches[pid]
        if n_cp < self.patch_out_cp:
            starts = [0]
        else:
            starts = [i * self.patch_out_cp for i in range(n_patches)]
            starts[-1] = max(0, n_cp - self.patch_out_cp)

        core_start_cp_raw = int(starts[patch_id])
        max_start = max(0, n_cp - self.patch_out_cp)

        if self.augment and self.jitter_cp > 0:
            if patch_id == 0:
                d = random.randint(-self.jitter_cp, self.jitter_cp)
                core_start_cp_raw = min(core_start_cp_raw + d, max_start)
                core_start_cp_raw = max(core_start_cp_raw, -(self.patch_out_cp - 1))
            elif 0 < patch_id < len(starts) - 1:
                d = random.randint(-self.jitter_cp, self.jitter_cp)
                core_start_cp_raw = min(max(0, core_start_cp_raw + d), max_start)

        in_start_cp_raw = int(core_start_cp_raw - self.halo_cp)

        pad_front_cp = max(0, -in_start_cp_raw)
        pad_front_cp = min(pad_front_cp, self.patch_in_cp - 1)
        start_cp = max(0, in_start_cp_raw)

        pad_front_px = pad_front_cp * self.cp_height_px
        need_px = window_px - pad_front_px
        start_px = start_cp * self.cp_height_px

        core_start_cp = int(self.halo_cp)
        core_end_cp = int(self.halo_cp + self.patch_out_cp)

        W_x = int(self.W_in) if self.W_in is not None else int(Wp)
        W_y = int(self.W)

        seg_x = drr[:, start_px: start_px + need_px, : min(W_x, Wp)]
        seg_y = sino[start_px: start_px + need_px, : min(W_y, sino.shape[1])]

        Hx, Wx = seg_x.shape[1], seg_x.shape[2]
        Hy, Wy = seg_y.shape[0], seg_y.shape[1]

        buf = np.zeros((C, window_px, W_x), dtype=np.float32)
        bufy = np.zeros((window_px, W_y), dtype=np.float32)

        buf[:, pad_front_px: pad_front_px + Hx, :Wx] = seg_x
        bufy[pad_front_px: pad_front_px + Hy, :Wy] = seg_y

        x_drr = _to_tensor_f32(buf)
        y_full = _to_tensor_f32(bufy).unsqueeze(0)

        cp_idx_raw = torch.arange(in_start_cp_raw, in_start_cp_raw + self.patch_in_cp, dtype=torch.float32)

        angles = _load_gantry_norm(
            info.get("gantry_file", None),
            n_cp_needed=self.patch_in_cp,
            start_cp=int(in_start_cp_raw),
            fallback_delta_deg=360.0 / 51.0,
        ).view(self.patch_in_cp, 1)

        den = max(n_cp - 1, 1)
        pos_idx = cp_idx_raw.clamp(min=0.0, max=float(max(n_cp - 1, 0)))
        positions = (pos_idx / den).clamp(min=0.0, max=1.0).view(self.patch_in_cp, 1)

        if self.augment:
            x_drr, y_full, angles = self._apply_pair_augs(x_drr, y_full, angles)
            x_drr = self._apply_channel_dropout(x_drr)

        x_drr = x_drr.clamp_(0.0, 1.0).contiguous()
        y_full = y_full.clamp_(0.0, 1.0).contiguous()
        y_sino = y_full[:, ::self.cp_height_px, :].contiguous()

        return _build_batch_dict(
            info,
            x_drr=x_drr,
            angles=angles.contiguous(),
            positions=positions.contiguous(),
            y_sino=y_sino,
            core_start_cp=core_start_cp,
            core_end_cp=core_end_cp,
        )


# =========================
# Dataset (val, full)
# =========================
class SinogramValDataset(Dataset):
    """
    Validation pleine séquence (ou pad jusqu'à patch_cp).
    Pas d'augmentation.
    """

    def __init__(
        self,
        subjects,
        W_out: int,
        W_in: int | None = None,
        cp_height_px: int = 8,
        cp_unit: int | None = None,
        patch_cp: int = 256,
        mmap: bool = True,
    ):
        self.subjects = subjects
        self.W = int(W_out)
        self.W_in = int(W_in) if W_in is not None else None
        if cp_unit is not None and int(cp_unit) != int(cp_height_px):
            raise ValueError(f"cp_unit ({cp_unit}) doit egaler cp_height_px ({cp_height_px})")
        self.cp_height_px = int(cp_height_px if cp_unit is None else cp_unit)
        self.cp_unit = self.cp_height_px
        self.patch_cp = int(patch_cp)
        self.mmap = bool(mmap)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        info = self.subjects[idx]


        drr = np.load(info["x_file"], mmap_mode="r") if self.mmap else np.load(info["x_file"])
        sino = np.load(info["y_file"], mmap_mode="r") if self.mmap else np.load(info["y_file"])
        drr = drr.astype(np.float32, copy=False)
        sino = sino.astype(np.float32, copy=False)

        C, Lpx, Wp = drr.shape
        n_cp_orig = Lpx // self.cp_height_px
        H_px = n_cp_orig * self.cp_height_px

        if sino.ndim == 2 and sino.shape[0] == n_cp_orig and Lpx == n_cp_orig * self.cp_unit:
            sino = np.repeat(sino, self.cp_height_px, axis=0)

        W_x = int(self.W_in) if self.W_in is not None else int(Wp)
        W_y = int(self.W)

        x = _to_tensor_f32(drr[:, :H_px, : min(W_x, Wp)])
        y_full = _to_tensor_f32(sino[:H_px, : min(W_y, sino.shape[1])]).unsqueeze(0)

        if n_cp_orig < self.patch_cp:
            pad_cp = self.patch_cp - n_cp_orig
            pad_px = pad_cp * self.cp_height_px
            x = F.pad(x, (0, 0, 0, pad_px), value=0.0)
            y_full = F.pad(y_full, (0, 0, 0, pad_px), value=0.0)
            n_cp = self.patch_cp
        else:
            n_cp = n_cp_orig

        angles = _load_gantry_norm(
            info.get("gantry_file", None),
            n_cp_needed=n_cp,
            start_cp=0,
            fallback_delta_deg=360.0 / 51.0,
        )

        cp_idx = torch.arange(n_cp, dtype=torch.float32)
        den = max(n_cp_orig - 1, 1)
        pos_idx = torch.clamp(cp_idx, max=float(max(n_cp_orig - 1, 0)))
        positions = (pos_idx / den).clamp(min=0.0, max=1.0).view(n_cp, 1)

        y_sino = y_full[:, ::self.cp_height_px, :]

        return _build_batch_dict(
            info,
            x_drr=x.clamp(0.0, 1.0).contiguous(),
            angles=angles.contiguous(),
            positions=positions.contiguous(),
            y_sino=y_sino.clamp(0.0, 1.0).contiguous(),
        )


def _resolve_loader_api(
    *,
    path: str | None,
    data_dir: str | None,
    W_out: int | None,
    W: int | None,
    cp_height_px: int | None,
    cp_unit: int | None,
) -> tuple[str, int | None, int | None]:
    if path and data_dir and path != data_dir:
        raise ValueError(f"path ({path}) et data_dir ({data_dir}) differents")
    root = path or data_dir
    if not root:
        raise ValueError("get_patch_loaders: fournir `path` (ou alias legacy `data_dir`)")

    if W_out is not None and W is not None and int(W_out) != int(W):
        raise ValueError(f"W_out ({W_out}) et alias W ({W}) differents")
    w_out_resolved = int(W_out if W_out is not None else W) if (W_out is not None or W is not None) else None

    if cp_height_px is not None and cp_unit is not None and int(cp_height_px) != int(cp_unit):
        raise ValueError(f"cp_height_px ({cp_height_px}) et alias cp_unit ({cp_unit}) differents")
    cp_resolved = int(cp_height_px if cp_height_px is not None else cp_unit) if (cp_height_px is not None or cp_unit is not None) else None

    if data_dir is not None:
        LOGGER.warning("`data_dir` est deprecie, utiliser `path`.")
    if W is not None:
        LOGGER.warning("`W` est deprecie, utiliser `W_out`.")
    if cp_unit is not None and cp_height_px is None:
        LOGGER.warning("`cp_unit` est deprecie, utiliser `cp_height_px`.")

    return root, w_out_resolved, cp_resolved


# =========================
# factory
# =========================
def get_patch_loaders(
    path: str | None = None,
    batch_size: int = 1,
    ratio: float = 0.9,
    *,
    data_dir: str | None = None,
    dose_min_gy: float,
    dose_max_gy: float,
    cp_dur_min_sec: float,
    cp_dur_max_sec: float,
    film_with_presence: bool = True,
    W_out: int | None = None,
    W_in: int | None = None,
    cp_height_px: int | None = None,
    W: int | None = None,
    cp_unit: int | None = None,
    patch_cp: int = 256,
    patch_in_cp: int | None = None,
    patch_out_cp: int | None = None,
    halo_cp: int = 0,
    jitter_cp: int = 32,
    seed: int = 42,
    augment: bool = True,
    num_workers: int = 0,
    mmap: bool = True,
    split_json: str | None = None,
):
    """Fabrique DataLoader train/val avec API stable (`path`, `W_out`, `cp_height_px`)."""
    import numpy as np

    path, W_out, cp_height_px = _resolve_loader_api(
        path=path,
        data_dir=data_dir,
        W_out=W_out,
        W=W,
        cp_height_px=cp_height_px,
        cp_unit=cp_unit,
    )
    cp_height_px = 8 if cp_height_px is None else int(cp_height_px)

    patch_out_cp = int(patch_out_cp) if patch_out_cp is not None else int(patch_cp)
    halo_cp = int(halo_cp)
    patch_in_cp = int(patch_in_cp) if patch_in_cp is not None else int(patch_out_cp + 2 * halo_cp)

    if split_json:
        allow_train = _load_split(split_json, "train")
        allow_val = _load_split(split_json, "val")

        tr = _collect_subjects(path, allowed_ids=allow_train,
                               dose_min_gy=dose_min_gy, dose_max_gy=dose_max_gy,
                               cp_dur_min_sec=cp_dur_min_sec, cp_dur_max_sec=cp_dur_max_sec,
                               film_with_presence=film_with_presence)
        va = _collect_subjects(path, allowed_ids=allow_val,
                               dose_min_gy=dose_min_gy, dose_max_gy=dose_max_gy,
                               cp_dur_min_sec=cp_dur_min_sec, cp_dur_max_sec=cp_dur_max_sec,
                               film_with_presence=film_with_presence)
        subs = tr + va
    else:
        subs = _collect_subjects(path, allowed_ids=None,
                                 dose_min_gy=dose_min_gy, dose_max_gy=dose_max_gy,
                                 cp_dur_min_sec=cp_dur_min_sec, cp_dur_max_sec=cp_dur_max_sec,
                                 film_with_presence=film_with_presence)
        random.seed(seed)
        random.shuffle(subs)
        n_train = int(len(subs) * ratio)
        tr, va = subs[:n_train], subs[n_train:]

    if W_out is None:
        Ws = [np.load(s["y_file"], mmap_mode="r").shape[1] for s in subs]
        W_out = int(max(Ws)) if len(Ws) else 64

    if W_in is None and len(subs) > 0:
        W_in = int(np.load(subs[0]["x_file"], mmap_mode="r").shape[2])

    ds_tr = SinogramPatchAugmentedDataset(
        tr,
        W_out=W_out,
        W_in=W_in,
        cp_height_px=cp_height_px,
        patch_in_cp=patch_in_cp,
        patch_out_cp=patch_out_cp,
        halo_cp=halo_cp,
        jitter_cp=jitter_cp,
        augment=augment,
        mmap=mmap,
    )
    ds_va = SinogramValDataset(
        va,
        W_out=W_out,
        W_in=W_in,
        cp_height_px=cp_height_px,
        patch_cp=patch_in_cp,
        mmap=mmap,
    )

    train_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def get_full_sequence_loader(
    split: str,
    path: str | None = None,
    batch_size: int = 1,
    ratio: float = 0.9,
    *,
    data_dir: str | None = None,
    dose_min_gy: float,
    dose_max_gy: float,
    cp_dur_min_sec: float,
    cp_dur_max_sec: float,
    film_with_presence: bool = True,
    W_out: int | None = None,
    W_in: int | None = None,
    cp_height_px: int | None = None,
    W: int | None = None,
    cp_unit: int | None = None,
    patch_cp: int = 256,
    patch_in_cp: int | None = None,
    patch_out_cp: int | None = None,
    halo_cp: int = 0,
    seed: int = 42,
    num_workers: int = 0,
    mmap: bool = True,
    split_json: str | None = None,
):
    """Fabrique un DataLoader pleine sequence (SinogramValDataset) pour train/val/test.

    Méthode 1: un patient complet par batch (batch_size=1) pour éviter tout padding inter-patients.
    """
    import numpy as np

    split = str(split).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split doit etre 'train'|'val'|'test', recu: {split}")
    if int(batch_size) != 1:
        raise ValueError(
            "get_full_sequence_loader impose batch_size=1 (un patient complet par batch, sans padding inter-patients)."
        )

    path, W_out, cp_height_px = _resolve_loader_api(
        path=path,
        data_dir=data_dir,
        W_out=W_out,
        W=W,
        cp_height_px=cp_height_px,
        cp_unit=cp_unit,
    )
    cp_height_px = 8 if cp_height_px is None else int(cp_height_px)
    patch_out_cp = int(patch_out_cp) if patch_out_cp is not None else int(patch_cp)
    halo_cp = int(halo_cp)
    patch_in_cp = int(patch_in_cp) if patch_in_cp is not None else int(patch_out_cp + 2 * halo_cp)

    if split_json:
        allowed_ids = _load_split(split_json, split)
        subjects = _collect_subjects(
            path,
            allowed_ids=allowed_ids,
            dose_min_gy=dose_min_gy,
            dose_max_gy=dose_max_gy,
            cp_dur_min_sec=cp_dur_min_sec,
            cp_dur_max_sec=cp_dur_max_sec,
            film_with_presence=film_with_presence,
        )
    else:
        subjects_all = _collect_subjects(
            path,
            allowed_ids=None,
            dose_min_gy=dose_min_gy,
            dose_max_gy=dose_max_gy,
            cp_dur_min_sec=cp_dur_min_sec,
            cp_dur_max_sec=cp_dur_max_sec,
            film_with_presence=film_with_presence,
        )
        random.seed(seed)
        random.shuffle(subjects_all)
        n_train = int(len(subjects_all) * ratio)
        train_subjects = subjects_all[:n_train]
        val_subjects = subjects_all[n_train:]

        if split == "train":
            subjects = train_subjects
        elif split == "val":
            subjects = val_subjects
        else:
            subjects = subjects_all

    if W_out is None:
        Ws = [np.load(s["y_file"], mmap_mode="r").shape[1] for s in subjects]
        W_out = int(max(Ws)) if len(Ws) else 64

    if W_in is None and len(subjects) > 0:
        W_in = int(np.load(subjects[0]["x_file"], mmap_mode="r").shape[2])

    ds = SinogramValDataset(
        subjects,
        W_out=W_out,
        W_in=W_in,
        cp_height_px=cp_height_px,
        patch_cp=patch_in_cp,
        mmap=mmap,
    )
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=(split == "train"),
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )


def get_full_train_loader(*args, **kwargs):
    """Alias explicite pour le stage-2 (train pleine sequence)."""
    kwargs = dict(kwargs)
    kwargs.pop("split", None)
    return get_full_sequence_loader("train", *args, **kwargs)


def get_test_loader(
    path: str | None = None,
    *,
    data_dir: str | None = None,
    dose_min_gy: float,
    dose_max_gy: float,
    cp_dur_min_sec: float,
    cp_dur_max_sec: float,
    film_with_presence: bool = True,
    W_out: int | None = None,
    W_in: int | None = None,
    cp_height_px: int | None = None,
    W: int | None = None,
    cp_unit: int | None = None,
    patch_cp: int = 256,
    patch_in_cp: int | None = None,
    patch_out_cp: int | None = None,
    halo_cp: int = 0,
    num_workers: int = 0,
    mmap: bool = True,
    split_json: str | None = None,
):
    import numpy as np

    path, W_out, cp_height_px = _resolve_loader_api(
        path=path,
        data_dir=data_dir,
        W_out=W_out,
        W=W,
        cp_height_px=cp_height_px,
        cp_unit=cp_unit,
    )
    cp_height_px = 8 if cp_height_px is None else int(cp_height_px)
    patch_out_cp = int(patch_out_cp) if patch_out_cp is not None else int(patch_cp)
    halo_cp = int(halo_cp)
    patch_in_cp = int(patch_in_cp) if patch_in_cp is not None else int(patch_out_cp + 2 * halo_cp)

    if split_json:
        allow_test = _load_split(split_json, "test")
    else:
        # In inference mode, allow evaluating all discovered subjects when no split manifest is provided.
        allow_test = None

    te = _collect_subjects(path, allowed_ids=allow_test,
                           dose_min_gy=dose_min_gy, dose_max_gy=dose_max_gy,
                           cp_dur_min_sec=cp_dur_min_sec, cp_dur_max_sec=cp_dur_max_sec,
                           film_with_presence=film_with_presence)

    if W_out is None:
        Ws = [np.load(s["y_file"], mmap_mode="r").shape[1] for s in te]
        W_out = int(max(Ws)) if len(Ws) else 64

    if W_in is None and len(te) > 0:
        W_in = int(np.load(te[0]["x_file"], mmap_mode="r").shape[2])

    ds_te = SinogramValDataset(te, W_out=W_out, W_in=W_in, cp_height_px=cp_height_px, patch_cp=patch_in_cp, mmap=mmap)
    test_loader = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )
    return test_loader
