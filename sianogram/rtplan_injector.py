#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtplan_injector.py  —  Plug & Play (SIA)

Objectif (par patient_id / ipp) :
  - prendre un sinogramme prédit (y_pred.npy / sino.npy) en REL (0..1) ou ABS (secondes/CP)
  - retrouver le RTPLAN dans dicom_data/<patient_id>/... (dossiers suffixés *_0, *_1, ...)
  - copier le plan original -> <test_root>/<patient_id>/RP_original.dcm
  - injecter le sinogramme -> <test_root>/<patient_id>/RP_injected.dcm

Arbo attendue côté test :
  <test_root>/<patient_id>/
    - y_pred.npy (ou sino.npy)
    - RP_original.dcm
    - RP_injected.dcm

Notes :
  - On s'aligne sur les ControlPointSequence qui portent le tag 300D,10A7 (sinogramme Tomo)
  - On déduit t_cp (sec/CP) :
        1) BeamMeterset (minutes) / N_CP
        2) fichier t_used_sec.npy dans le répertoire patient (optionnel)
        3) défaut 18/51 s
  - Si entrée ABS -> convertit en REL
  - Nettoyage ABS (sur y_abs = y_rel * t_cp) :
        - 0 <= y_abs < 60 ms        => 0
        - y_abs > t_cp - 60 ms      => 1
  - PrimaryDosimeterUnit='MINUTE' + BeamMeterset cohérent
  - Désapprouve le plan et peut renommer (option)

Dépendances :
  pip install pydicom numpy
"""

import os
import shutil
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pydicom
from pydicom.valuerep import DSfloat
from pydicom.dataelem import DataElement


# -------------------------
# DICOM tags
# -------------------------
TAG_PRIMARY_UNIT = (0x300A, 0x00B3)          # PrimaryDosimeterUnit
TAG_BEAM_METERSET = (0x300A, 0x0086)         # BeamMeterset (in FractionGroupSequence/ReferencedBeamSequence)
TAG_APPROVAL_STATUS = (0x300E, 0x0002)       # ApprovalStatus
TAG_REVIEW_DATE = (0x300E, 0x0008)
TAG_REVIEW_TIME = (0x300E, 0x0009)
TAG_REVIEWER_NAME = (0x300E, 0x000A)
TAG_RT_PLAN_LABEL = (0x300A, 0x0002)         # RTPlanLabel
TAG_RT_PLAN_NAME = (0x300A, 0x0003)          # RTPlanName
TAG_RT_PLAN_DATE = (0x300A, 0x0006)          # RTPlanDate
TAG_RT_PLAN_TIME = (0x300A, 0x0007)          # RTPlanTime
TAG_TREATMENT_MACHINE_NAME = (0x300A, 0x00B2)
TAG_SINO = (0x300D, 0x10A7)                  # Tomo sinogram private tag used in your files


# -------------------------
# Default temporal params
# -------------------------
DEFAULT_GANTRY_PERIOD_SEC = 18.0
DEFAULT_CP_PER_TURN = 51
DEFAULT_T_CP_SEC = DEFAULT_GANTRY_PERIOD_SEC / DEFAULT_CP_PER_TURN  # ≈ 0.35294 s

# -------------------------
# SIA-specific defaults
# -------------------------
SIANOGRAM_PLAN_PREFIX = "sIAnogram_"          # Plan name prefix → "sIAnogram_YYYYMMDD"
SIANOGRAM_MACHINE_NAME = "Rdx_1_dble_calc"    # Machine name for all beams

# Opening thresholds (0 = disabled / ne pas modifier l'histogramme)
DEFAULT_LOW_THRESH_SEC  = 0.020   # 20 ms  → valeurs < seuil mises à 0
DEFAULT_HIGH_THRESH_SEC = 0.000   # 0 ms   → désactivé par défaut


# -------------------------
# Options
# -------------------------
@dataclass
class InjectorConfig:
    """Full configuration for RTPLAN sinogram injection."""

    # ---------- Input ----------
    # "rel": fractions in [0..1]  |  "abs": seconds per CP
    pred_input_mode: str = "rel"

    # ---------- Width ----------
    force_resample: bool = True
    default_width: int = 64

    # ---------- Temporal fallback ----------
    default_t_cp_sec: float = DEFAULT_T_CP_SEC

    # ---------- Opening thresholds ----------
    # low_thresh_sec  : valeurs en-dessous mises à 0.  Si == 0 → pas de modification.
    # high_thresh_sec : valeurs au-dessus de (t_cp - seuil) mises à t_cp.  Si == 0 → pas de modification.
    low_thresh_sec:  float = DEFAULT_LOW_THRESH_SEC   # 20 ms
    high_thresh_sec: float = DEFAULT_HIGH_THRESH_SEC  # 0 ms = désactivé

    # ---------- Dosimetry ----------
    set_primary_unit_minute: bool = True
    set_beam_meterset_minutes: bool = True

    # ---------- Plan naming ----------
    # Le plan sera nommé <rename_prefix><YYYYMMDD>  (ex: "sIAnogram_20260319")
    rename_plan: bool = True
    rename_prefix: str = SIANOGRAM_PLAN_PREFIX

    # ---------- Approval ----------
    deapprove: bool = True

    # ---------- Machine ----------
    set_machine_name: bool = True
    machine_name: str = SIANOGRAM_MACHINE_NAME

    # ---------- Output filenames ----------
    out_rp_original: str = "RP_original.dcm"
    out_rp_injected: str = "RP_injected.dcm"


# =========================
# Patient folder resolution
# =========================

def _pick_patient_dir(dicom_root: Path, patient_id: str) -> Path:
    """
    Retourne le dossier patient DICOM correspondant à patient_id.

    Tolère :
      - '341144_0' (exact)
      - '341144'   -> essaie '341144_0.._9'
      - '3411440'  -> essaie '341144_0' (heuristique "underscore manquant")
      - prefix match '341144_*'
    """
    dicom_root = Path(dicom_root)
    pid = str(patient_id).strip()

    # 0) exact
    direct = dicom_root / pid
    if direct.is_dir():
        return direct

    # 1) heuristique : "underscore manquant" (ex: 3118480 -> 311848_0)
    if pid.isdigit() and len(pid) >= 2:
        cand = dicom_root / f"{pid[:-1]}_{pid[-1]}"
        if cand.is_dir():
            return cand

    # 2) essais _0.._9
    for k in range(10):
        cand = dicom_root / f"{pid}_{k}"
        if cand.is_dir():
            return cand

    # 3) match prefix pid_*
    matches: List[Path] = sorted([p for p in dicom_root.glob(f"{pid}_*") if p.is_dir()])
    if matches:
        # préférence *_0
        for p in matches:
            if p.name.endswith("_0"):
                return p
        return matches[0]

    # 4) si pid contient déjà "_" on tente base
    if "_" in pid:
        base = pid.split("_")[0]
        direct2 = dicom_root / base
        if direct2.is_dir():
            return direct2
        for k in range(10):
            cand = dicom_root / f"{base}_{k}"
            if cand.is_dir():
                return cand
        matches = sorted([p for p in dicom_root.glob(f"{base}_*") if p.is_dir()])
        if matches:
            for p in matches:
                if p.name.endswith("_0"):
                    return p
            return matches[0]

    raise FileNotFoundError(f"Dossier patient introuvable: {dicom_root/pid}")

# =========================
# Low-level helpers
# =========================
def _robust_load_sino(sino_path: Path) -> np.ndarray:
    """
    Charge y_pred.npy / sino.npy et normalise en [N_CP, W] float32.
    Accepte :
      - [N_CP, W]
      - [1, N_CP, W]
      - [1, 1, N_CP, W]
    """
    y = np.load(sino_path)
    y = np.asarray(y, dtype=np.float32)

    if y.ndim == 4 and y.shape[0] == 1:
        y = y[0]
    if y.ndim == 3 and y.shape[0] == 1:
        y = y[0]
    if y.ndim != 2:
        raise ValueError(f"format inattendu: shape={y.shape} (attendu 2D [N_CP, W])")

    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    return y


def resample_columns(arr: np.ndarray, target_w: int) -> np.ndarray:
    ncp, w = arr.shape
    if w == target_w:
        return arr
    x_src = np.linspace(0.0, 1.0, w, dtype=np.float32)
    x_tgt = np.linspace(0.0, 1.0, target_w, dtype=np.float32)
    out = np.empty((ncp, target_w), dtype=np.float32)
    for i in range(ncp):
        out[i] = np.interp(x_tgt, x_src, arr[i])
    return out.astype(np.float32, copy=False)


def set_primary_unit_minute(ds: pydicom.Dataset) -> None:
    try:
        if hasattr(ds, "PrimaryDosimeterUnit"):
            ds.PrimaryDosimeterUnit = "MINUTE"
        else:
            ds.add_new(TAG_PRIMARY_UNIT, "CS", "MINUTE")
    except Exception:
        pass


def get_beam_meterset_minutes(ds: pydicom.Dataset) -> Optional[float]:
    try:
        fgs = getattr(ds, "FractionGroupSequence", None)
        if not fgs:
            return None
        for fg in fgs:
            rbs = getattr(fg, "ReferencedBeamSequence", None)
            if not rbs:
                continue
            for rb in rbs:
                v = getattr(rb, "BeamMeterset", None)
                if v is not None:
                    return float(v)
    except Exception:
        return None
    return None


def set_beam_meterset_minutes(ds: pydicom.Dataset, meterset_minutes: float) -> bool:
    try:
        fgs = getattr(ds, "FractionGroupSequence", None)
        if not fgs:
            return False
        for fg in fgs:
            rbs = getattr(fg, "ReferencedBeamSequence", None)
            if not rbs:
                continue
            for rb in rbs:
                rb.BeamMeterset = float(meterset_minutes)
        return True
    except Exception:
        return False


def set_plan_name(ds: pydicom.Dataset, prefix: str = SIANOGRAM_PLAN_PREFIX) -> None:
    """Set RTPlanLabel and RTPlanName to <prefix><YYYYMMDD> (e.g. 'sIAnogram_20260319')."""
    today = datetime.date.today().strftime("%Y%m%d")
    new = f"{prefix}{today}"
    for tag, vr, attr in [
        (TAG_RT_PLAN_LABEL, "LO", "RTPlanLabel"),
        (TAG_RT_PLAN_NAME,  "LO", "RTPlanName"),
    ]:
        try:
            if hasattr(ds, attr):
                setattr(ds, attr, new)
            else:
                ds.add_new(tag, vr, new)
        except Exception:
            pass


# backward-compat alias
def rename_plan_with_today(ds: pydicom.Dataset, suffix: str = "sianogram") -> None:
    """Deprecated alias — calls set_plan_name(prefix=suffix)."""
    set_plan_name(ds, prefix=suffix)


def deapprove_plan(ds: pydicom.Dataset) -> None:
    try:
        if hasattr(ds, "ApprovalStatus"):
            ds.ApprovalStatus = "UNAPPROVED"
        else:
            ds.add_new(TAG_APPROVAL_STATUS, "CS", "UNAPPROVED")
    except Exception:
        pass
    for tag in (TAG_REVIEW_DATE, TAG_REVIEW_TIME, TAG_REVIEWER_NAME):
        try:
            if tag in ds:
                del ds[tag]
        except Exception:
            pass


def set_all_beams_machine_name(ds: pydicom.Dataset, machine_name: str = "Tomo3") -> None:
    try:
        if not hasattr(ds, "BeamSequence") or not ds.BeamSequence:
            return
        for beam in ds.BeamSequence:
            try:
                if hasattr(beam, "TreatmentMachineName"):
                    beam.TreatmentMachineName = machine_name
                else:
                    beam.add_new(TAG_TREATMENT_MACHINE_NAME, "SH", machine_name)
            except Exception:
                continue
    except Exception:
        pass


# =========================
# RTPLAN resolution
# =========================
def find_rtplan_for_patient(dicom_root: Path, patient_id: str) -> Path:
    dicom_root = Path(dicom_root)
    base = _pick_patient_dir(dicom_root, patient_id)

    # 1) RP* en priorité
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.upper().startswith("RP"):
                return Path(root) / fn

    # 2) fallback: Modality RTPLAN
    for root, _, files in os.walk(base):
        for fn in files:
            p = Path(root) / fn
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                if getattr(ds, "Modality", "").upper() == "RTPLAN":
                    # Si PatientID existe, il peut être base (sans suffix) -> on ne bloque pas
                    return p
            except Exception:
                continue

    raise FileNotFoundError(f"Aucun RTPLAN trouvé pour patient_id={patient_id} sous {base}")


# =========================
# Sino tag handling
# =========================
def extract_cp_keep_indices(ds: pydicom.Dataset) -> List[int]:
    cps = ds.BeamSequence[0].ControlPointSequence
    keep = [k for k, cp in enumerate(cps) if TAG_SINO in cp]
    if not keep:
        raise RuntimeError("Aucun ControlPoint ne contient 300D,10A7 dans ce RTPLAN.")
    return keep


def detect_sino_encoding(ds: pydicom.Dataset) -> Tuple[str, str, Optional[int]]:
    """
    Retourne (VR, mode, width)
      mode:
        - "ascii"    : bytes/str avec séparateur '\\'
        - "textlist" : list[DS]
        - "f32"      : bytes float32
    """
    cps = ds.BeamSequence[0].ControlPointSequence
    for cp in cps:
        if TAG_SINO in cp:
            elem = cp[TAG_SINO]
            vr = elem.VR
            val = elem.value

            if isinstance(val, (bytes, bytearray)):
                if b"\\" in val:
                    try:
                        w = len(val.decode("ascii", "ignore").split("\\"))
                    except Exception:
                        w = None
                    return vr or "UN", "ascii", w
                nbytes = len(val)
                w = nbytes // 4 if (nbytes % 4 == 0) else None
                return vr or "OB", "f32", w

            if isinstance(val, (list, tuple)):
                return vr or "DS", "textlist", len(val)

            if isinstance(val, str):
                return vr or "DS", "ascii", len(val.split("\\"))

            return vr or "UN", "ascii", None

    return "UN", "ascii", None


def resolve_target_width(ds: pydicom.Dataset, default_width: int = 64) -> int:
    _, _, w = detect_sino_encoding(ds)
    return int(w) if (w is not None and w > 0) else int(default_width)


def derive_t_cp_seconds(
    n_cp: int,
    ds: pydicom.Dataset,
    patient_out_dir: Path,
    default_t_cp_sec: float,
) -> Tuple[float, str]:
    """
    1) BeamMeterset(min)/n_cp
    2) t_used_sec.npy (optionnel) dans patient_out_dir
    3) default_t_cp_sec
    """
    ms_min = get_beam_meterset_minutes(ds)
    if (ms_min is not None) and np.isfinite(ms_min) and ms_min > 0 and n_cp > 0:
        t_cp = float(ms_min) * 60.0 / float(n_cp)
        if np.isfinite(t_cp) and t_cp > 0:
            return t_cp, "rtplan_beam_meterset"

    t_path = patient_out_dir / "t_used_sec.npy"
    if t_path.exists():
        try:
            arr = np.load(t_path).astype(np.float32).reshape(-1)
            if arr.size >= 1 and np.isfinite(arr[0]) and arr[0] > 0:
                return float(arr[0]), "file_t_used_sec"
        except Exception:
            pass

    return float(default_t_cp_sec), "default_t_cp"


def clean_sino_with_abs_seconds(
    y_rel: np.ndarray,
    t_cp_sec: float,
    low_thresh_sec: float,
    high_thresh_sec: float,
) -> np.ndarray:
    """
    Nettoie le sinogramme REL en appliquant les seuils d'ouverture.

    Règles (en secondes absolues y_abs = y_rel * t_cp) :
      - si low_thresh_sec  > 0 : y_abs < low_thresh_sec            → 0
      - si high_thresh_sec > 0 : y_abs > t_cp - high_thresh_sec    → t_cp
      - si thresh == 0          : ce seuil n'est PAS appliqué
    """
    if not np.isfinite(t_cp_sec) or t_cp_sec <= 0:
        raise ValueError(f"t_cp_sec invalide: {t_cp_sec}")

    y_abs = y_rel.astype(np.float32, copy=True) * float(t_cp_sec)

    if float(low_thresh_sec) > 0.0:
        np.putmask(y_abs, (y_abs >= 0.0) & (y_abs < float(low_thresh_sec)), 0.0)

    if float(high_thresh_sec) > 0.0:
        high_thr = max(0.0, float(t_cp_sec) - float(high_thresh_sec))
        np.putmask(y_abs, y_abs > high_thr, float(t_cp_sec))

    y_rel_clean = y_abs / float(t_cp_sec)
    np.clip(y_rel_clean, 0.0, 1.0, out=y_rel_clean)
    return y_rel_clean.astype(np.float32, copy=False)


def write_sinogram_into_rtplan(ds: pydicom.Dataset, sino_rows_rel: np.ndarray, force_last_zero: bool = True) -> None:
    """
    Ecrit le sinogramme REL (0..1) dans 300D,10A7 en respectant l'encodage détecté.
    """
    assert sino_rows_rel.ndim == 2, sino_rows_rel.shape
    ncp, W = sino_rows_rel.shape

    if force_last_zero and ncp >= 1:
        sino_rows_rel = sino_rows_rel.copy()
        sino_rows_rel[-1, :] = 0.0

    vr0, mode0, w0 = detect_sino_encoding(ds)
    if (w0 is not None) and (int(w0) != int(W)):
        raise ValueError(f"Largeur W incompatible: pred={W} vs RTPLAN={w0}")

    vr_to_use = vr0 or "OB"
    mode_to_use = mode0 or "f32"

    cps = ds.BeamSequence[0].ControlPointSequence
    keep = [k for k, cp in enumerate(cps) if TAG_SINO in cp]

    if keep and len(keep) != ncp:
        raise RuntimeError(f"Nombre de CP porteurs du tag ({len(keep)}) != N_CP prédits ({ncp})")

    for i in range(ncp):
        row = sino_rows_rel[i].astype(np.float32, copy=False)

        if mode_to_use == "ascii":
            payload = ("\\".join(f"{float(v):.6f}" for v in row)).encode("ascii", "ignore")
            vr_write = vr_to_use if vr_to_use in ("UN", "OB") else "UN"

        elif mode_to_use == "textlist":
            payload = [DSfloat(v) for v in row]
            vr_write = "DS"

        else:  # "f32"
            payload = row.tobytes()
            vr_write = "OB"

        cp_target = cps[keep[i]] if keep else cps[i]

        if TAG_SINO in cp_target:
            elem = cp_target[TAG_SINO]
            if elem.VR != vr_write:
                cp_target[TAG_SINO] = DataElement(TAG_SINO, vr_write, payload)
            else:
                elem.value = payload
                cp_target[TAG_SINO] = elem
        else:
            cp_target.add_new(TAG_SINO, vr_write, payload)


# =========================
# Public API
# =========================
def inject_sino_for_patient(
    patient_id: str,
    sino_path: Path,
    dicom_root: Path,
    out_patient_dir: Path,
    cfg: Optional[InjectorConfig] = None,
) -> Dict[str, Any]:
    """
    Pipeline complet (1 patient).
    Renvoie un dict report (ok/ko, chemins, etc.).
    """
    cfg = cfg or InjectorConfig()
    out_patient_dir = Path(out_patient_dir)
    dicom_root = Path(dicom_root)
    sino_path = Path(sino_path)

    out_patient_dir.mkdir(parents=True, exist_ok=True)

    if not sino_path.exists():
        return {"ok": False, "patient_id": patient_id, "reason": f"fichier sino introuvable: {sino_path}"}

    # 1) load sino
    y_in = _robust_load_sino(sino_path)
    n_cp_pred, w_pred = y_in.shape

    # 2) RTPLAN
    rt_src = find_rtplan_for_patient(dicom_root, patient_id)
    ds = pydicom.dcmread(str(rt_src))

    # 3) check N_CP on CP that contain TAG_SINO
    keep = extract_cp_keep_indices(ds)
    n_cp_rt = len(keep)
    if n_cp_rt != n_cp_pred:
        return {
            "ok": False,
            "patient_id": patient_id,
            "reason": f"Mismatch N_CP pred={n_cp_pred} vs RTPLAN(tagged_CP)={n_cp_rt}",
            "rtplan_in": str(rt_src),
        }

    # 4) width
    target_w = resolve_target_width(ds, default_width=cfg.default_width)
    if cfg.force_resample and (w_pred != target_w):
        y_in = resample_columns(y_in, target_w)

    # 5) t_cp
    t_cp_sec, t_cp_src = derive_t_cp_seconds(
        n_cp_pred, ds, out_patient_dir, default_t_cp_sec=cfg.default_t_cp_sec
    )

    # 6) convert abs->rel if needed
    if str(cfg.pred_input_mode).lower() == "abs":
        y_rel = np.clip(y_in / float(t_cp_sec), 0.0, 1.0)
    else:
        y_rel = np.clip(y_in, 0.0, 1.0)

    # 7) clean
    y_rel_clean = clean_sino_with_abs_seconds(
        y_rel,
        t_cp_sec=float(t_cp_sec),
        low_thresh_sec=cfg.low_thresh_sec,
        high_thresh_sec=cfg.high_thresh_sec,
    )

    # 8) write sino
    write_sinogram_into_rtplan(ds, y_rel_clean, force_last_zero=True)

    # 9) update units/meterset
    meterset_minutes = (float(t_cp_sec) * float(n_cp_pred)) / 60.0
    if cfg.set_primary_unit_minute:
        set_primary_unit_minute(ds)
    if cfg.set_beam_meterset_minutes:
        set_beam_meterset_minutes(ds, meterset_minutes)

    # 10) plan meta
    if cfg.rename_plan:
        set_plan_name(ds, prefix=cfg.rename_prefix)
    if cfg.deapprove:
        deapprove_plan(ds)
    if cfg.set_machine_name:
        set_all_beams_machine_name(ds, machine_name=cfg.machine_name)

    # 11) outputs
    rp_original = out_patient_dir / cfg.out_rp_original
    rp_injected = out_patient_dir / cfg.out_rp_injected

    try:
        shutil.copy2(rt_src, rp_original)
    except Exception:
        pass

    ds.save_as(str(rp_injected))

    return {
        "ok": True,
        "patient_id": patient_id,
        "rtplan_in": str(rt_src),
        "rp_original": str(rp_original),
        "rp_injected": str(rp_injected),
        "sino_in": str(sino_path),
        "n_cp": int(n_cp_pred),
        "width": int(y_rel_clean.shape[1]),
        "t_cp_sec": float(t_cp_sec),
        "t_cp_source": t_cp_src,
        "beam_meterset_minutes": float(meterset_minutes),
        "pred_input_mode": str(cfg.pred_input_mode).lower(),
    }


def inject_from_test_root(
    test_root: Path,
    dicom_root: Path,
    cfg: Optional[InjectorConfig] = None,
    pred_filenames: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Batch : itère sur test_root/<patient_id>/, et injecte en cherchant un fichier prédiction.
    Par défaut, cherche: y_pred.npy puis sino.npy.
    """
    test_root = Path(test_root)
    dicom_root = Path(dicom_root)
    cfg = cfg or InjectorConfig()

    if pred_filenames is None:
        pred_filenames = ["y_pred.npy", "sino.npy"]  # ordre de priorité

    reports: List[Dict[str, Any]] = []
    for pdir in sorted([p for p in test_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        # Ignore internal/hidden folders created by runtime tooling.
        if pdir.name.startswith("_") or pdir.name.startswith("."):
            continue
        patient_id = pdir.name

        sino_path = None
        for fn in pred_filenames:
            cand = pdir / fn
            if cand.exists():
                sino_path = cand
                break

        if sino_path is None:
            reports.append({
                "ok": False,
                "patient_id": patient_id,
                "reason": f"Aucun fichier trouvé parmi {pred_filenames} dans {pdir}"
            })
            continue

        rep = inject_sino_for_patient(
            patient_id=patient_id,
            sino_path=sino_path,
            dicom_root=dicom_root,
            out_patient_dir=pdir,
            cfg=cfg,
        )
        reports.append(rep)

    return reports


def run_injector(
    dicom_root: str,
    test_root: str,
    mode: str = "rel",
    pred_filenames: Optional[List[str]] = None,
    rename: bool = True,
    rename_prefix: str = SIANOGRAM_PLAN_PREFIX,
    set_machine: bool = True,
    machine_name: str = SIANOGRAM_MACHINE_NAME,
    low_thresh_sec: float = DEFAULT_LOW_THRESH_SEC,
    high_thresh_sec: float = DEFAULT_HIGH_THRESH_SEC,
    no_resample: bool = False,
) -> List[Dict[str, Any]]:
    """
    Appel direct depuis PyCharm, sans CLI.
    """
    cfg = InjectorConfig(
        pred_input_mode=mode,
        force_resample=(not no_resample),
        rename_plan=bool(rename),
        rename_prefix=str(rename_prefix),
        set_machine_name=bool(set_machine),
        machine_name=str(machine_name),
        low_thresh_sec=float(low_thresh_sec),
        high_thresh_sec=float(high_thresh_sec),
    )

    reports = inject_from_test_root(
        test_root=Path(test_root),
        dicom_root=Path(dicom_root),
        cfg=cfg,
        pred_filenames=pred_filenames,
    )

    ok = sum(1 for r in reports if r.get("ok"))
    ko = len(reports) - ok
    print(f"[RTPLAN_INJECT] Done. OK={ok} KO={ko}")
    for r in reports:
        if not r.get("ok"):
            print(f"  - KO {r.get('patient_id')}: {r.get('reason')}")
    return reports


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inject predicted sinograms into RTPLAN DICOM files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rtplan_injector.py \\
    --dicom-root /mnt/LeGrosDisque/Julien/sianogramme/dicom_data \\
    --test-root inference_outputs/20260319_120000__best

  python rtplan_injector.py \\
    --dicom-root /data/dicom_data \\
    --test-root inference_outputs/run1 \\
    --low-thresh 0.020 --high-thresh 0.060 \\
    --report-json injection_report.json
        """,
    )

    parser.add_argument("--dicom-root", required=True, help="Root folder containing patient DICOM directories")
    parser.add_argument("--test-root", required=True, help="Root folder containing one subfolder per patient")
    parser.add_argument("--mode", choices=["rel", "abs"], default="rel", help="Interpretation of prediction values")
    parser.add_argument(
        "--pred-filenames", nargs="+", default=["y_pred.npy", "sino.npy"],
        help="Prediction file names searched in each patient directory",
    )
    parser.add_argument(
        "--low-thresh", type=float, default=DEFAULT_LOW_THRESH_SEC,
        help=f"Low opening threshold (s). Values below → 0. 0 = disabled. (default: {DEFAULT_LOW_THRESH_SEC})",
    )
    parser.add_argument(
        "--high-thresh", type=float, default=DEFAULT_HIGH_THRESH_SEC,
        help=f"High opening threshold (s). Values above t_cp-thresh → t_cp. 0 = disabled. (default: {DEFAULT_HIGH_THRESH_SEC})",
    )
    parser.add_argument(
        "--no-rename", action="store_true",
        help="Disable plan renaming (default: rename to sIAnogram_YYYYMMDD)",
    )
    parser.add_argument(
        "--rename-prefix", default=SIANOGRAM_PLAN_PREFIX,
        help=f"Plan name prefix (default: '{SIANOGRAM_PLAN_PREFIX}')",
    )
    parser.add_argument(
        "--no-set-machine", action="store_true",
        help="Disable machine name override (default: set to Rdx_1_dble_calc)",
    )
    parser.add_argument(
        "--machine-name", default=SIANOGRAM_MACHINE_NAME,
        help=f"Machine name (default: '{SIANOGRAM_MACHINE_NAME}')",
    )
    parser.add_argument("--no-resample", action="store_true", help="Disable width resampling")
    parser.add_argument("--report-json", default=None, help="Optional path to save reports as JSON")

    return parser


def main() -> None:
    import json

    parser = _build_arg_parser()
    args = parser.parse_args()

    reports = run_injector(
        dicom_root=args.dicom_root,
        test_root=args.test_root,
        mode=args.mode,
        pred_filenames=args.pred_filenames,
        rename=(not args.no_rename),
        rename_prefix=args.rename_prefix,
        set_machine=(not args.no_set_machine),
        machine_name=args.machine_name,
        low_thresh_sec=args.low_thresh,
        high_thresh_sec=args.high_thresh,
        no_resample=args.no_resample,
    )

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)
        print(f"[RTPLAN_INJECT] Report written to: {args.report_json}")


if __name__ == "__main__":
    main()
