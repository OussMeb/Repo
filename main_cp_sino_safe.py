#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_cp_sino.py

Single-file baseline for CP-wise Tomo/Radixact sinogram generation.

Dataset assumption:
  patient_folder/
    CT*.dcm  # many slices
    RD*.dcm  # one RTDOSE
    RP*.dcm  # one RTPLAN with Tomo private sinogram tag (300D,10A7)
    RS*.dcm  # optional for this first CT+RTDOSE baseline

Formulation:
  CT + RTDOSE + safe RP geometry at CP t -> sinogram[t, 64]

RP sinogram values are used only as supervised targets, never as input features.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

try:
    from scipy.ndimage import map_coordinates, rotate, shift, zoom
except Exception as exc:
    raise RuntimeError("Missing scipy. Install with: pip install scipy") from exc


DEFAULT_RAW_ROOT = Path("/home/oussama/Desktop/Here/corrected")
DEFAULT_OUT_ROOT = Path("./processed_cp_sino")
TOMO_SINO_TAG = (0x300D, 0x10A7)
LEAF_COUNT = 64
EPS = 1e-8


def require_pydicom() -> Any:
    if pydicom is None:
        raise RuntimeError("Missing pydicom. Install it with: pip install pydicom")
    return pydicom


def dcmread(path: str | Path, **kwargs: Any) -> Any:
    return require_pydicom().dcmread(str(path), **kwargs)


@dataclass(frozen=True)
class CTVolume:
    array_hu_zyx: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]
    origin_xyz_mm: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class RTPlanData:
    rtplan_path: str
    sino: np.ndarray
    angles_deg: np.ndarray
    table_mm: np.ndarray
    table_attr: str
    cumulative_meterset_weight: np.ndarray
    cp_duration_sec: np.ndarray
    beam_meterset_minutes: float
    isocenter_xyz_mm: tuple[float, float, float]
    n_cp: int


@dataclass(frozen=True)
class PatientBundle:
    patient_id: str
    patient_dir: str
    ct_path_count: int
    rd_path: str
    rp_path: str
    rs_path: str | None
    n_cp: int
    sino_shape: tuple[int, int]
    cp_view_shape: tuple[int, int, int, int] | None


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root.addHandler(handler)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def find_first(patient_dir: Path, prefix: str) -> Path | None:
    files = sorted(patient_dir.glob(f"{prefix}*.dcm"))
    return files[0] if files else None


def find_ct_files(patient_dir: Path) -> list[Path]:
    return sorted(patient_dir.glob("CT*.dcm"))


def scan_patient_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    return sorted([p for p in raw_root.iterdir() if p.is_dir()])


def get_first_beam(ds: pydicom.Dataset) -> Any:
    if hasattr(ds, "BeamSequence") and len(ds.BeamSequence) > 0:
        return ds.BeamSequence[0]
    tag = (0x300A, 0x00B0)
    if tag in ds and len(ds[tag].value) > 0:
        return ds[tag].value[0]
    raise RuntimeError("RTPLAN has no readable BeamSequence")


def get_control_points(beam: Any) -> list[Any]:
    if hasattr(beam, "ControlPointSequence"):
        return list(beam.ControlPointSequence)
    tag = (0x300A, 0x0111)
    if tag in beam:
        return list(beam[tag].value)
    raise RuntimeError("Beam has no readable ControlPointSequence")


def read_ct_volume(ct_files: Sequence[Path]) -> CTVolume:
    if not ct_files:
        raise FileNotFoundError("No CT*.dcm files found")

    slice_rows = []
    for path in ct_files:
        ds = dcmread(str(path), force=True)
        if not hasattr(ds, "ImagePositionPatient"):
            raise RuntimeError(f"CT slice missing ImagePositionPatient: {path}")
        ipp = [float(v) for v in ds.ImagePositionPatient]
        slice_rows.append((ipp[2], ds))

    slice_rows.sort(key=lambda x: x[0])
    datasets = [ds for _, ds in slice_rows]

    arrays = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        arrays.append(arr * slope + intercept)

    volume = np.stack(arrays, axis=0).astype(np.float32)
    first = datasets[0]
    last = datasets[-1]

    row_spacing = safe_float(first.PixelSpacing[0], 1.0)
    col_spacing = safe_float(first.PixelSpacing[1], 1.0)
    if len(datasets) > 1:
        dz = (safe_float(last.ImagePositionPatient[2]) - safe_float(first.ImagePositionPatient[2])) / (len(datasets) - 1)
        dz = abs(dz) if abs(dz) > EPS else safe_float(getattr(first, "SliceThickness", 1.0), 1.0)
    else:
        dz = safe_float(getattr(first, "SliceThickness", 1.0), 1.0)

    return CTVolume(
        array_hu_zyx=volume,
        spacing_zyx_mm=(float(dz), float(row_spacing), float(col_spacing)),
        origin_xyz_mm=tuple(float(v) for v in first.ImagePositionPatient),
        shape_zyx=tuple(int(v) for v in volume.shape),
    )


def dose_z_coordinates(ds: pydicom.Dataset, dose_nz: int) -> np.ndarray:
    dose_ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float32)

    if hasattr(ds, "GridFrameOffsetVector"):
        offsets = np.asarray([float(v) for v in ds.GridFrameOffsetVector], dtype=np.float32)
        if offsets.size == dose_nz:
            if offsets.size > 0 and abs(float(offsets[0] - dose_ipp[2])) < 1e-3:
                return offsets.astype(np.float32)
            return (dose_ipp[2] + offsets).astype(np.float32)

    dz = safe_float(getattr(ds, "SliceThickness", 1.0), 1.0)
    return (dose_ipp[2] + np.arange(dose_nz, dtype=np.float32) * dz).astype(np.float32)


def resample_dose_to_ct_grid(rd_path: Path, ct: CTVolume, chunk_z: int = 8) -> np.ndarray:
    ds = dcmread(str(rd_path), force=True)
    dose = ds.pixel_array.astype(np.float32)
    if dose.ndim == 2:
        dose = dose[None, :, :]
    dose *= safe_float(getattr(ds, "DoseGridScaling", 1.0), 1.0)

    if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "PixelSpacing"):
        raise RuntimeError(f"RTDOSE missing geometry tags: {rd_path}")

    dose_ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float32)
    dose_row_spacing = safe_float(ds.PixelSpacing[0], 1.0)
    dose_col_spacing = safe_float(ds.PixelSpacing[1], 1.0)

    dose_nz, _, _ = dose.shape
    z_coords = dose_z_coordinates(ds, dose_nz)
    if dose_nz > 1:
        dose_dz = float((z_coords[-1] - z_coords[0]) / (dose_nz - 1))
        if abs(dose_dz) < EPS:
            dose_dz = safe_float(getattr(ds, "SliceThickness", 1.0), 1.0)
    else:
        dose_dz = safe_float(getattr(ds, "SliceThickness", 1.0), 1.0)

    ct_nz, ct_ny, ct_nx = ct.shape_zyx
    ct_dz, ct_dy, ct_dx = ct.spacing_zyx_mm
    ct_origin_x, ct_origin_y, ct_origin_z = ct.origin_xyz_mm

    ct_x = ct_origin_x + np.arange(ct_nx, dtype=np.float32) * ct_dx
    ct_y = ct_origin_y + np.arange(ct_ny, dtype=np.float32) * ct_dy
    ct_z = ct_origin_z + np.arange(ct_nz, dtype=np.float32) * ct_dz

    dose_x_idx = (ct_x - dose_ipp[0]) / max(dose_col_spacing, EPS)
    dose_y_idx = (ct_y - dose_ipp[1]) / max(dose_row_spacing, EPS)
    dose_z_idx = (ct_z - float(z_coords[0])) / dose_dz

    out = np.zeros((ct_nz, ct_ny, ct_nx), dtype=np.float32)
    for z0 in range(0, ct_nz, max(1, int(chunk_z))):
        z1 = min(z0 + max(1, int(chunk_z)), ct_nz)
        zz, yy, xx = np.meshgrid(dose_z_idx[z0:z1], dose_y_idx, dose_x_idx, indexing="ij")
        coords = np.asarray([zz, yy, xx], dtype=np.float32)
        out[z0:z1] = map_coordinates(
            dose,
            coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def parse_sino_value(value: Any) -> list[float] | None:
    if isinstance(value, (bytes, bytearray)):
        text = value.decode(errors="ignore")
        parts = [p for p in text.split("\\") if p.strip()]
    elif isinstance(value, str):
        parts = [p for p in value.split("\\") if p.strip()]
    else:
        try:
            row = [float(x) for x in value]
            return row if len(row) == LEAF_COUNT else None
        except Exception:
            return None

    try:
        row = [float(p) for p in parts]
    except ValueError:
        return None
    return row if len(row) == LEAF_COUNT else None


def extract_sinogram_from_cps(cps: Iterable[Any]) -> np.ndarray:
    rows = []
    for cp in cps:
        if TOMO_SINO_TAG not in cp:
            continue
        row = parse_sino_value(cp[TOMO_SINO_TAG].value)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No Tomo sinogram found in RP private tag (300D,10A7)")

    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def extract_beam_meterset_minutes(ds: pydicom.Dataset) -> float:
    try:
        if hasattr(ds, "FractionGroupSequence") and len(ds.FractionGroupSequence) > 0:
            fgs = ds.FractionGroupSequence[0]
            if hasattr(fgs, "ReferencedBeamSequence") and len(fgs.ReferencedBeamSequence) > 0:
                rb = fgs.ReferencedBeamSequence[0]
                if hasattr(rb, "BeamMeterset"):
                    return float(rb.BeamMeterset)
    except Exception:
        pass

    try:
        beam = get_first_beam(ds)
        if hasattr(beam, "BeamMeterset"):
            return float(beam.BeamMeterset)
    except Exception:
        pass

    return 0.0


def extract_table_positions(cps: Sequence[Any]) -> tuple[np.ndarray, str]:
    attrs = ["TableTopLongitudinalPosition", "TableTopLateralPosition", "TableTopVerticalPosition"]
    candidates = {}
    for attr in attrs:
        candidates[attr] = np.asarray([safe_float(getattr(cp, attr, 0.0), 0.0) for cp in cps], dtype=np.float32)

    best_attr = attrs[0]
    best_range = -1.0
    for attr, arr in candidates.items():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        r = float(finite.max() - finite.min())
        if r > best_range:
            best_range = r
            best_attr = attr

    return candidates[best_attr], best_attr


def extract_isocenter_xyz(cps: Sequence[Any]) -> tuple[float, float, float]:
    if not cps:
        return (0.0, 0.0, 0.0)

    cp0 = cps[0]
    for attr in ("IsocenterPosition", "ISOCenterPosition"):
        if hasattr(cp0, attr):
            try:
                return tuple(float(v) for v in getattr(cp0, attr))
            except Exception:
                pass

    tag = (0x300A, 0x012C)
    if tag in cp0:
        try:
            return tuple(float(v) for v in cp0[tag].value)
        except Exception:
            pass

    return (0.0, 0.0, 0.0)


def extract_rtplan_data(rp_path: Path) -> RTPlanData:
    ds = dcmread(str(rp_path), stop_before_pixels=True, force=True)
    beam = get_first_beam(ds)
    cps = get_control_points(beam)

    sino = extract_sinogram_from_cps(cps)
    angles = np.asarray([safe_float(getattr(cp, "GantryAngle", 0.0), 0.0) for cp in cps], dtype=np.float32)
    table, table_attr = extract_table_positions(cps)
    cmw = np.asarray([safe_float(getattr(cp, "CumulativeMetersetWeight", 0.0), 0.0) for cp in cps], dtype=np.float32)

    beam_minutes = extract_beam_meterset_minutes(ds)
    if beam_minutes > 0 and len(cps) > 0:
        dcmw = np.diff(cmw, prepend=0.0)
        dcmw = np.clip(dcmw, 0.0, None)
        if float(dcmw.sum()) <= EPS:
            cp_duration = np.full((len(cps),), beam_minutes * 60.0 / len(cps), dtype=np.float32)
        else:
            cp_duration = (beam_minutes * 60.0 * dcmw / dcmw.sum()).astype(np.float32)
    else:
        cp_duration = np.zeros((len(cps),), dtype=np.float32)

    if sino.shape[0] != len(cps):
        n = min(sino.shape[0], len(cps))
        logging.warning("Sinogram rows and CP count differ for %s: sino=%s cps=%s. Cropping to %s.",
                        rp_path, sino.shape[0], len(cps), n)
        sino = sino[:n]
        angles = angles[:n]
        table = table[:n]
        cmw = cmw[:n]
        cp_duration = cp_duration[:n]

    return RTPlanData(
        rtplan_path=str(rp_path),
        sino=sino.astype(np.float32, copy=False),
        angles_deg=angles.astype(np.float32, copy=False),
        table_mm=table.astype(np.float32, copy=False),
        table_attr=table_attr,
        cumulative_meterset_weight=cmw.astype(np.float32, copy=False),
        cp_duration_sec=cp_duration.astype(np.float32, copy=False),
        beam_meterset_minutes=float(beam_minutes),
        isocenter_xyz_mm=extract_isocenter_xyz(cps),
        n_cp=int(sino.shape[0]),
    )


def downsample_volume(volume: np.ndarray, factors_zyx: tuple[float, float, float], order: int) -> np.ndarray:
    zoom_factors = tuple(1.0 / max(float(v), 1.0) for v in factors_zyx)
    if all(abs(v - 1.0) < EPS for v in zoom_factors):
        return volume.astype(np.float32, copy=False)
    return zoom(volume, zoom_factors, order=order).astype(np.float32, copy=False)


def normalize_ct_hu(volume: np.ndarray, lo: float = -1000.0, hi: float = 1000.0) -> np.ndarray:
    arr = np.clip(volume.astype(np.float32), lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def normalize_dose_gy(dose: np.ndarray, dose_norm_gy: float = 70.0) -> np.ndarray:
    arr = np.nan_to_num(dose.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if dose_norm_gy <= 0:
        finite = arr[np.isfinite(arr)]
        dose_norm_gy = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        dose_norm_gy = max(dose_norm_gy, 1.0)
    return np.clip(arr / float(dose_norm_gy), 0.0, 2.0).astype(np.float32)


def resize_2d(arr: np.ndarray, out_h: int, out_w: int, order: int = 1) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"resize_2d expects 2D array, got shape={arr.shape}")

    out = zoom(arr, (out_h / max(arr.shape[0], 1), out_w / max(arr.shape[1], 1)), order=order).astype(np.float32)
    if out.shape != (out_h, out_w):
        fixed = np.zeros((out_h, out_w), dtype=np.float32)
        h = min(out_h, out.shape[0])
        w = min(out_w, out.shape[1])
        fixed[:h, :w] = out[:h, :w]
        out = fixed
    return out


def project_rotated_volume(volume_zyx: np.ndarray,
                           angle_deg: float,
                           z_shift_vox: float,
                           out_h: int,
                           out_w: int,
                           reducer: str,
                           interp_order: int) -> np.ndarray:
    shifted = shift(volume_zyx, shift=(float(z_shift_vox), 0.0, 0.0),
                    order=interp_order, mode="constant", cval=0.0, prefilter=False)

    rotated = rotate(shifted, angle=-float(angle_deg), axes=(1, 2), reshape=False,
                     order=interp_order, mode="constant", cval=0.0, prefilter=False)

    if reducer == "max":
        proj = rotated.max(axis=1)
    elif reducer == "sum":
        proj = rotated.sum(axis=1) / float(max(rotated.shape[1], 1))
    elif reducer == "mean":
        proj = rotated.mean(axis=1)
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")

    return resize_2d(np.nan_to_num(proj, nan=0.0), out_h=out_h, out_w=out_w, order=1)


def build_cp_features(plan: RTPlanData, cp_indices: np.ndarray) -> np.ndarray:
    cp_indices = cp_indices.astype(np.int64, copy=False)

    angles_rad = np.deg2rad(plan.angles_deg[cp_indices].astype(np.float32))
    sin_a = np.sin(angles_rad)
    cos_a = np.cos(angles_rad)

    table = plan.table_mm[cp_indices].astype(np.float32)
    table_norm = (table - float(np.mean(plan.table_mm))) / max(float(np.std(plan.table_mm)), 1.0)

    cp_norm = cp_indices.astype(np.float32) / max(float(plan.n_cp - 1), 1.0)

    if np.any(plan.cp_duration_sec > 0):
        dur_mean = float(np.mean(plan.cp_duration_sec[plan.cp_duration_sec > 0]))
    else:
        dur_mean = 1.0
    dur_norm = plan.cp_duration_sec[cp_indices].astype(np.float32) / max(dur_mean, EPS)

    return np.stack([sin_a, cos_a, table_norm, cp_norm, dur_norm], axis=1).astype(np.float32)


def generate_cp_views(ct_norm: np.ndarray,
                      dose_norm: np.ndarray,
                      ct_spacing_zyx_mm: tuple[float, float, float],
                      plan: RTPlanData,
                      view_height: int,
                      leaf_count: int,
                      cp_stride: int,
                      max_cp: int | None,
                      table_axis: str,
                      ct_reducer: str,
                      dose_reducer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = max(1, int(cp_stride))
    cp_indices = np.arange(plan.n_cp, dtype=np.int32)[::step]
    if max_cp is not None:
        cp_indices = cp_indices[: int(max_cp)]

    views = np.zeros((len(cp_indices), 2, int(view_height), int(leaf_count)), dtype=np.float32)
    table_ref = float(plan.table_mm[0]) if plan.table_mm.size else 0.0
    z_spacing = float(ct_spacing_zyx_mm[0])

    for out_i, cp_i in enumerate(cp_indices):
        if table_axis == "z":
            z_shift_vox = -float(plan.table_mm[int(cp_i)] - table_ref) / max(z_spacing, EPS)
        elif table_axis == "none":
            z_shift_vox = 0.0
        else:
            raise ValueError(f"Unsupported table_axis: {table_axis}")

        angle = float(plan.angles_deg[int(cp_i)])
        views[out_i, 0] = project_rotated_volume(ct_norm, angle, z_shift_vox, view_height, leaf_count, ct_reducer, 1)
        views[out_i, 1] = project_rotated_volume(dose_norm, angle, z_shift_vox, view_height, leaf_count, dose_reducer, 1)

        if (out_i + 1) % 50 == 0 or out_i == 0 or out_i + 1 == len(cp_indices):
            logging.info("  CP views: %d/%d", out_i + 1, len(cp_indices))

    return cp_indices, views, build_cp_features(plan, cp_indices)


def save_preview_png(out_path: Path, cp_views: np.ndarray, sino_rows: np.ndarray, max_panels: int = 6) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        logging.warning("matplotlib unavailable; skipping preview.")
        return

    n = min(max_panels, cp_views.shape[0])
    if n <= 0:
        return

    selected = np.linspace(0, cp_views.shape[0] - 1, n).astype(int)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3 * n))
    if n == 1:
        axes = np.asarray([axes])

    for row_i, idx in enumerate(selected):
        axes[row_i, 0].imshow(cp_views[idx, 0], aspect="auto", vmin=0.0, vmax=1.0)
        axes[row_i, 0].set_title(f"CT CP-view #{idx}")
        axes[row_i, 1].imshow(cp_views[idx, 1], aspect="auto", vmin=0.0, vmax=1.2)
        axes[row_i, 1].set_title(f"Dose CP-view #{idx}")
        axes[row_i, 2].plot(sino_rows[idx])
        axes[row_i, 2].set_ylim(-0.05, 1.05)
        axes[row_i, 2].set_title("Target sino row [64]")

    for ax in axes.reshape(-1):
        ax.grid(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_sinogram_png(out_path: Path, sino: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sino, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Leaf index")
    ax.set_ylabel("Control point")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def preprocess_one_patient(patient_dir: Path,
                           out_root: Path,
                           save_views: bool,
                           save_volumes: bool,
                           view_height: int,
                           leaf_count: int,
                           xy_downsample: float,
                           z_downsample: float,
                           dose_norm_gy: float,
                           cp_stride: int,
                           max_cp: int | None,
                           table_axis: str,
                           ct_reducer: str,
                           dose_reducer: str,
                           preview: bool) -> PatientBundle:
    patient_id = patient_dir.name
    patient_out = out_root / patient_id
    patient_out.mkdir(parents=True, exist_ok=True)

    ct_files = find_ct_files(patient_dir)
    rd_path = find_first(patient_dir, "RD")
    rp_path = find_first(patient_dir, "RP")
    rs_path = find_first(patient_dir, "RS")

    if not ct_files:
        raise FileNotFoundError(f"{patient_id}: missing CT*.dcm")
    if rd_path is None:
        raise FileNotFoundError(f"{patient_id}: missing RD*.dcm")
    if rp_path is None:
        raise FileNotFoundError(f"{patient_id}: missing RP*.dcm")

    logging.info("[%s] reading CT (%d slices)", patient_id, len(ct_files))
    ct = read_ct_volume(ct_files)

    logging.info("[%s] reading/resampling RD", patient_id)
    dose_on_ct = resample_dose_to_ct_grid(rd_path, ct)

    logging.info("[%s] extracting RP sinogram target", patient_id)
    plan = extract_rtplan_data(rp_path)

    factors = (float(z_downsample), float(xy_downsample), float(xy_downsample))
    ct_small = downsample_volume(ct.array_hu_zyx, factors, order=1)
    dose_small = downsample_volume(dose_on_ct, factors, order=1)
    spacing_small = (
        ct.spacing_zyx_mm[0] * max(float(z_downsample), 1.0),
        ct.spacing_zyx_mm[1] * max(float(xy_downsample), 1.0),
        ct.spacing_zyx_mm[2] * max(float(xy_downsample), 1.0),
    )

    ct_norm = normalize_ct_hu(ct_small)
    dose_norm = normalize_dose_gy(dose_small, dose_norm_gy=dose_norm_gy)

    np.save(patient_out / "sino.npy", plan.sino.astype(np.float32))
    np.save(patient_out / "angles_deg.npy", plan.angles_deg.astype(np.float32))
    np.save(patient_out / "table_mm.npy", plan.table_mm.astype(np.float32))
    np.save(patient_out / "cp_duration_sec.npy", plan.cp_duration_sec.astype(np.float32))
    np.save(patient_out / "cumulative_meterset_weight.npy", plan.cumulative_meterset_weight.astype(np.float32))
    save_sinogram_png(patient_out / "sino.png", plan.sino, title=f"{patient_id} target sinogram")

    if save_volumes:
        np.save(patient_out / "ct_norm_downsampled.npy", ct_norm.astype(np.float32))
        np.save(patient_out / "dose_norm_downsampled.npy", dose_norm.astype(np.float32))

    step = max(1, int(cp_stride))
    cp_indices = np.arange(plan.n_cp, dtype=np.int32)[::step]
    if max_cp is not None:
        cp_indices = cp_indices[: int(max_cp)]

    features = build_cp_features(plan, cp_indices)
    np.save(patient_out / "cp_indices.npy", cp_indices)
    np.save(patient_out / "cp_features.npy", features)

    cp_view_shape = None
    if save_views:
        logging.info("[%s] generating CP views", patient_id)
        cp_indices, cp_views, features = generate_cp_views(
            ct_norm=ct_norm,
            dose_norm=dose_norm,
            ct_spacing_zyx_mm=spacing_small,
            plan=plan,
            view_height=view_height,
            leaf_count=leaf_count,
            cp_stride=cp_stride,
            max_cp=max_cp,
            table_axis=table_axis,
            ct_reducer=ct_reducer,
            dose_reducer=dose_reducer,
        )
        np.save(patient_out / "cp_indices.npy", cp_indices)
        np.save(patient_out / "cp_features.npy", features)
        np.save(patient_out / "cp_views.npy", cp_views)
        cp_view_shape = tuple(int(v) for v in cp_views.shape)

        if preview:
            save_preview_png(patient_out / "preview_cp_views.png", cp_views, plan.sino[cp_indices])

    meta = {
        "patient_id": patient_id,
        "patient_dir": str(patient_dir),
        "ct_files_count": len(ct_files),
        "rd_path": str(rd_path),
        "rp_path": str(rp_path),
        "rs_path": str(rs_path) if rs_path else None,
        "ct_shape_zyx": ct.shape_zyx,
        "ct_spacing_zyx_mm": ct.spacing_zyx_mm,
        "ct_downsampled_shape_zyx": tuple(int(v) for v in ct_small.shape),
        "ct_downsampled_spacing_zyx_mm": spacing_small,
        "dose_max_gy_original_grid": float(np.max(dose_on_ct)) if dose_on_ct.size else 0.0,
        "rtplan": {
            "n_cp": plan.n_cp,
            "sino_shape": tuple(int(v) for v in plan.sino.shape),
            "table_attr_used": plan.table_attr,
            "beam_meterset_minutes": plan.beam_meterset_minutes,
            "isocenter_xyz_mm": plan.isocenter_xyz_mm,
            "angle_minmax_deg": [float(np.min(plan.angles_deg)), float(np.max(plan.angles_deg))],
            "table_minmax_mm": [float(np.min(plan.table_mm)), float(np.max(plan.table_mm))],
        },
        "cp_dataset": {
            "save_views": save_views,
            "save_volumes": save_volumes,
            "cp_stride": cp_stride,
            "max_cp": max_cp,
            "view_height": view_height,
            "leaf_count": leaf_count,
            "channels": ["ct_projection", "dose_projection"],
            "cp_view_shape": cp_view_shape,
            "feature_columns": ["sin_angle", "cos_angle", "table_norm", "cp_index_norm", "duration_norm"],
            "table_axis": table_axis,
            "ct_reducer": ct_reducer,
            "dose_reducer": dose_reducer,
            "target_rule": "sino[cp_index, :] from RP private tag (300D,10A7)",
            "leakage_note": "RP private sinogram values are target only and are not used as input features.",
        },
    }
    write_json(patient_out / "metadata.json", meta)

    return PatientBundle(
        patient_id=patient_id,
        patient_dir=str(patient_dir),
        ct_path_count=len(ct_files),
        rd_path=str(rd_path),
        rp_path=str(rp_path),
        rs_path=str(rs_path) if rs_path else None,
        n_cp=plan.n_cp,
        sino_shape=tuple(int(v) for v in plan.sino.shape),
        cp_view_shape=cp_view_shape,
    )


def command_inspect(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    patient_dirs = scan_patient_dirs(raw_root)
    if args.limit is not None:
        patient_dirs = patient_dirs[: int(args.limit)]

    rows = []
    for patient_dir in patient_dirs:
        ct_files = find_ct_files(patient_dir)
        rd = find_first(patient_dir, "RD")
        rp = find_first(patient_dir, "RP")
        rs = find_first(patient_dir, "RS")

        info = {
            "patient_id": patient_dir.name,
            "patient_dir": str(patient_dir),
            "n_ct": len(ct_files),
            "rd_path": str(rd) if rd else None,
            "rp_path": str(rp) if rp else None,
            "rs_path": str(rs) if rs else None,
            "has_rd": rd is not None,
            "has_rp": rp is not None,
            "has_rs": rs is not None,
            "n_cp": None,
            "sino_shape": None,
            "angle_minmax": None,
            "table_minmax": None,
            "table_attr": None,
            "status": "ok",
        }

        if rp is not None:
            try:
                plan = extract_rtplan_data(rp)
                info["n_cp"] = plan.n_cp
                info["sino_shape"] = list(plan.sino.shape)
                info["angle_minmax"] = [float(np.min(plan.angles_deg)), float(np.max(plan.angles_deg))]
                info["table_minmax"] = [float(np.min(plan.table_mm)), float(np.max(plan.table_mm))]
                info["table_attr"] = plan.table_attr
            except Exception as exc:
                info["status"] = f"rp_error: {exc}"

        rows.append(info)
        print(
            f"{info['patient_id']:>20} | CT={info['n_ct']:>4} | RD={info['has_rd']} | "
            f"RP={info['has_rp']} | RS={info['has_rs']} | N_CP={info['n_cp']} | "
            f"sino={info['sino_shape']} | {info['status']}"
        )

    if args.out_json:
        write_json(Path(args.out_json), {"raw_root": str(raw_root), "patients": rows})


def command_preprocess(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    patient_dirs = scan_patient_dirs(raw_root)
    if args.limit is not None:
        patient_dirs = patient_dirs[: int(args.limit)]

    bundles = []
    failures = []

    for i, patient_dir in enumerate(patient_dirs, 1):
        logging.info("Processing %d/%d: %s", i, len(patient_dirs), patient_dir.name)
        try:
            bundle = preprocess_one_patient(
                patient_dir=patient_dir,
                out_root=out_root,
                save_views=bool(args.save_views),
                save_volumes=bool(args.save_volumes),
                view_height=int(args.view_height),
                leaf_count=int(args.leaf_count),
                xy_downsample=float(args.xy_downsample),
                z_downsample=float(args.z_downsample),
                dose_norm_gy=float(args.dose_norm_gy),
                cp_stride=int(args.cp_stride),
                max_cp=args.max_cp,
                table_axis=str(args.table_axis),
                ct_reducer=str(args.ct_reducer),
                dose_reducer=str(args.dose_reducer),
                preview=bool(args.preview),
            )
            bundles.append(asdict(bundle))
        except Exception as exc:
            logging.exception("Failed patient %s: %s", patient_dir.name, exc)
            failures.append({"patient_id": patient_dir.name, "error": str(exc)})

    write_json(out_root / "preprocess_report.json", {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "n_success": len(bundles),
        "n_failed": len(failures),
        "success": bundles,
        "failures": failures,
    })


class CPViewDatasetWrapper:
    """Lazy PyTorch dataset over precomputed patient CP views."""

    def __init__(self, patient_dirs: list[Path]):
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self, dirs: list[Path]):
                self.dirs = dirs
                self.entries: list[tuple[int, int]] = []
                self.cache: dict[int, dict[str, Any]] = {}

                for d_i, d in enumerate(dirs):
                    if not (d / "cp_indices.npy").exists() or not (d / "cp_views.npy").exists():
                        continue
                    cp_indices = np.load(d / "cp_indices.npy")
                    for local_i in range(len(cp_indices)):
                        self.entries.append((d_i, local_i))

            def _load_dir(self, d_i: int) -> dict[str, Any]:
                if d_i not in self.cache:
                    d = self.dirs[d_i]
                    self.cache[d_i] = {
                        "views": np.load(d / "cp_views.npy", mmap_mode="r"),
                        "sino": np.load(d / "sino.npy", mmap_mode="r"),
                        "indices": np.load(d / "cp_indices.npy", mmap_mode="r"),
                        "features": np.load(d / "cp_features.npy", mmap_mode="r"),
                    }
                return self.cache[d_i]

            def __len__(self) -> int:
                return len(self.entries)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                d_i, local_i = self.entries[idx]
                data = self._load_dir(d_i)
                cp_i = int(data["indices"][local_i])

                x = np.array(data["views"][local_i], dtype=np.float32, copy=True)
                geom = np.array(data["features"][local_i], dtype=np.float32, copy=True)
                y = np.array(data["sino"][cp_i], dtype=np.float32, copy=True)

                return {
                    "x": torch.from_numpy(x),
                    "geom": torch.from_numpy(geom),
                    "y": torch.from_numpy(y),
                }

        self.dataset = _Dataset(patient_dirs)



def get_processed_patient_dirs(data_root: Path) -> list[Path]:
    """Return sample folders that contain precomputed CP views."""
    patient_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and (p / "cp_views.npy").exists()])
    if not patient_dirs:
        raise FileNotFoundError(f"No precomputed cp_views.npy found under {data_root}")
    return patient_dirs


def base_patient_id(sample_name: str) -> str:
    """
    Group repeated plan folders from the same patient.

    Examples:
        247887_0 -> 247887
        247887_1 -> 247887
        62547_0  -> 62547
    """
    parts = sample_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return sample_name


def split_patient_dirs_by_base_id(
    data_root: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path], dict[str, Any]]:
    """
    Split by base patient ID, not by CP and not by sample folder.

    This prevents folders like patient_0 and patient_1 from appearing in different splits.
    """
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be >= 0")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    sample_dirs = get_processed_patient_dirs(data_root)

    groups: dict[str, list[Path]] = {}
    for sample_dir in sample_dirs:
        groups.setdefault(base_patient_id(sample_dir.name), []).append(sample_dir)

    base_ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(base_ids)

    n_base = len(base_ids)
    if n_base == 1:
        train_ids, val_ids, test_ids = base_ids, [], []
    else:
        n_test = int(round(n_base * test_ratio))
        n_val = int(round(n_base * val_ratio))

        if test_ratio > 0 and n_base >= 3:
            n_test = max(1, n_test)
        if val_ratio > 0 and n_base >= 3:
            n_val = max(1, n_val)

        while n_test + n_val >= n_base:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        test_ids = base_ids[:n_test]
        val_ids = base_ids[n_test:n_test + n_val]
        train_ids = base_ids[n_test + n_val:]

        if not train_ids:
            raise RuntimeError("Patient split produced an empty training set.")

    def collect(ids: list[str]) -> list[Path]:
        out: list[Path] = []
        for pid in ids:
            out.extend(sorted(groups[pid]))
        return sorted(out)

    train_dirs = collect(train_ids)
    val_dirs = collect(val_ids)
    test_dirs = collect(test_ids)

    summary = {
        "n_base_patients": n_base,
        "n_sample_folders": len(sample_dirs),
        "train_base_ids": train_ids,
        "val_base_ids": val_ids,
        "test_base_ids": test_ids,
        "train_samples": [p.name for p in train_dirs],
        "val_samples": [p.name for p in val_dirs],
        "test_samples": [p.name for p in test_dirs],
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
    }
    return train_dirs, val_dirs, test_dirs, summary


def write_split_files(out_dir: Path, split_summary: dict[str, Any]) -> None:
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        base_ids = split_summary.get(f"{split}_base_ids", [])
        samples = split_summary.get(f"{split}_samples", [])
        (split_dir / f"{split}_patients.txt").write_text("\n".join(base_ids) + ("\n" if base_ids else ""), encoding="utf-8")
        (split_dir / f"{split}_samples.txt").write_text("\n".join(samples) + ("\n" if samples else ""), encoding="utf-8")

    write_json(split_dir / "split_summary.json", split_summary)


def load_patient_dirs_from_samples_file(data_root: Path, samples_file: Path) -> list[Path]:
    names = [x.strip() for x in samples_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    dirs = [data_root / name for name in names]
    missing = [str(p) for p in dirs if not (p / "cp_views.npy").exists()]
    if missing:
        raise FileNotFoundError(f"Some samples from {samples_file} are missing cp_views.npy: {missing[:5]}")
    return dirs

def build_model(view_channels: int = 2, geom_dim: int = 5, leaf_count: int = LEAF_COUNT):
    import torch
    import torch.nn as nn

    class CPViewSinoNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(view_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.geom = nn.Sequential(
                nn.Linear(geom_dim, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
            )
            self.head = nn.Sequential(
                nn.Linear(160, 128),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(128, leaf_count),
            )

        def forward(self, x: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
            h = self.conv(x).flatten(1)
            g = self.geom(geom)
            return torch.sigmoid(self.head(torch.cat([h, g], dim=1)))

    return CPViewSinoNet()


def sino_row_loss(pred, target, lambda_bg: float = 4.0, lambda_grad: float = 0.2):
    import torch.nn.functional as F

    target = target.clamp(0.0, 1.0)
    pred = pred.clamp(0.0, 1.0)

    weight = 1.0 + 2.0 * target.pow(2.0)
    base = (weight * F.smooth_l1_loss(pred, target, reduction="none")).mean()

    bg = (target <= 1e-4).to(pred.dtype)
    bg_loss = (bg * pred.pow(2.0)).mean()

    grad_p = pred[:, 1:] - pred[:, :-1]
    grad_t = target[:, 1:] - target[:, :-1]
    grad_loss = (grad_p - grad_t).abs().mean()

    return base + lambda_bg * bg_loss + lambda_grad * grad_loss




def compute_sino_metrics(
    pred_sino: np.ndarray,
    true_sino: np.ndarray,
    open_threshold: float = 0.01,
) -> dict[str, float]:
    """Sinogram-level metrics for one patient."""
    pred = np.asarray(pred_sino, dtype=np.float32)
    true = np.asarray(true_sino, dtype=np.float32)

    n_cp = min(pred.shape[0], true.shape[0])
    n_leaf = min(pred.shape[1], true.shape[1])
    pred = np.clip(pred[:n_cp, :n_leaf], 0.0, 1.0)
    true = np.clip(true[:n_cp, :n_leaf], 0.0, 1.0)

    err = pred - true
    abs_err = np.abs(err)

    true_open = true > float(open_threshold)
    pred_open = pred > float(open_threshold)
    true_closed = ~true_open

    tp = float(np.logical_and(pred_open, true_open).sum())
    fp = float(np.logical_and(pred_open, true_closed).sum())
    fn = float(np.logical_and(~pred_open, true_open).sum())

    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)

    energy_true = true.sum(axis=1)
    energy_pred = pred.sum(axis=1)

    metrics = {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(abs_err.max()) if abs_err.size else 0.0,
        "open_mae": float(abs_err[true_open].mean()) if np.any(true_open) else 0.0,
        "closed_mae": float(abs_err[true_closed].mean()) if np.any(true_closed) else 0.0,
        "background_leak_mean": float(pred[true_closed].mean()) if np.any(true_closed) else 0.0,
        "background_leak_p95": float(np.percentile(pred[true_closed], 95)) if np.any(true_closed) else 0.0,
        "background_leak_p99": float(np.percentile(pred[true_closed], 99)) if np.any(true_closed) else 0.0,
        "false_open_rate": float(np.logical_and(pred_open, true_closed).mean()) if true_closed.size else 0.0,
        "open_precision": float(precision),
        "open_recall": float(recall),
        "open_f1": float(f1),
        "true_open_fraction": float(true_open.mean()),
        "pred_open_fraction": float(pred_open.mean()),
        "energy_cp_mae": float(np.mean(np.abs(energy_pred - energy_true))),
        "energy_cp_rel_mae": float(np.mean(np.abs(energy_pred - energy_true) / np.maximum(energy_true, EPS))),
        "leaf_gradient_mae": float(np.mean(np.abs(np.diff(pred, axis=1) - np.diff(true, axis=1)))) if n_leaf > 1 else 0.0,
        "cp_gradient_mae": float(np.mean(np.abs(np.diff(pred, axis=0) - np.diff(true, axis=0)))) if n_cp > 1 else 0.0,
        "n_cp": float(n_cp),
        "n_leaf": float(n_leaf),
    }
    return metrics


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_patients": 0}

    metric_keys = sorted(
        key for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in {"n_cp", "n_leaf"}
    )
    summary: dict[str, Any] = {"n_patients": len(rows)}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if values.size:
            summary[f"macro_mean_{key}"] = float(values.mean())
            summary[f"macro_median_{key}"] = float(np.median(values))
    summary["total_cp"] = int(sum(float(row.get("n_cp", 0.0)) for row in rows))
    return summary


def predict_patient_with_model(
    model: Any,
    patient_dir: Path,
    device: Any,
    batch_size: int,
    open_threshold: float,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    ds = CPViewDatasetWrapper([patient_dir]).dataset
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

    cp_indices = np.load(patient_dir / "cp_indices.npy").astype(np.int32)
    true_sino = np.load(patient_dir / "sino.npy").astype(np.float32)
    pred_sampled = np.zeros_like(true_sino, dtype=np.float32)

    offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            geom = batch["geom"].to(device)
            pred = model(x, geom).detach().cpu().numpy().astype(np.float32)
            n = pred.shape[0]
            pred_sampled[cp_indices[offset: offset + n]] = pred
            offset += n

    pred_full = interpolate_missing_sino_rows(pred_sampled, cp_indices)

    full_metrics = compute_sino_metrics(pred_full, true_sino, open_threshold=open_threshold)
    sampled_metrics = compute_sino_metrics(pred_full[cp_indices], true_sino[cp_indices], open_threshold=open_threshold)

    row: dict[str, Any] = {
        "patient_id": patient_dir.name,
        "base_patient_id": base_patient_id(patient_dir.name),
        "n_cp": int(true_sino.shape[0]),
        "n_predicted_cp": int(len(cp_indices)),
    }
    row.update({f"full_{k}": v for k, v in full_metrics.items()})
    row.update({f"sampled_{k}": v for k, v in sampled_metrics.items()})

    if out_dir is not None:
        patient_out = out_dir / patient_dir.name
        patient_out.mkdir(parents=True, exist_ok=True)
        np.save(patient_out / "pred_sino.npy", pred_full)
        np.save(patient_out / "pred_sino_sampled_only.npy", pred_sampled)
        np.save(patient_out / "true_sino.npy", true_sino)
        save_sinogram_png(patient_out / "pred_sino.png", pred_full, title=f"{patient_dir.name} predicted sinogram")
        save_sinogram_png(patient_out / "true_sino.png", true_sino, title=f"{patient_dir.name} true sinogram")
        write_json(patient_out / "prediction_metrics.json", row)

    return row


def evaluate_model_on_dirs(
    model: Any,
    patient_dirs: list[Path],
    device: Any,
    batch_size: int,
    open_threshold: float,
    split_name: str,
    out_root: Path | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pred_root = None
    if out_root is not None:
        pred_root = out_root / f"{split_name}_predictions"
        pred_root.mkdir(parents=True, exist_ok=True)

    for patient_dir in patient_dirs:
        row = predict_patient_with_model(
            model=model,
            patient_dir=patient_dir,
            device=device,
            batch_size=batch_size,
            open_threshold=open_threshold,
            out_dir=pred_root,
        )
        rows.append(row)
        logging.info(
            "[%s] %s | full_MAE=%.6f full_F1=%.4f bg_p99=%.6f",
            split_name,
            patient_dir.name,
            float(row["full_mae"]),
            float(row["full_open_f1"]),
            float(row["full_background_leak_p99"]),
        )

    return {
        "split": split_name,
        "summary": aggregate_metric_rows(rows),
        "per_patient": rows,
    }

def command_train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dirs, val_dirs, test_dirs, split_summary = split_patient_dirs_by_base_id(
        data_root=data_root,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    write_split_files(out_dir, split_summary)

    train_ds = CPViewDatasetWrapper(train_dirs).dataset
    val_ds = CPViewDatasetWrapper(val_dirs).dataset if val_dirs else None

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty. Run preprocess with --save-views first.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available() and not args.cpu,
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(view_channels=2, geom_dim=5, leaf_count=int(args.leaf_count)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_score = float("inf")
    history: list[dict[str, Any]] = []

    logging.info(
        "patient-level split | train base=%d val base=%d test base=%d | train samples=%d val samples=%d test samples=%d | train CP=%d",
        len(split_summary["train_base_ids"]),
        len(split_summary["val_base_ids"]),
        len(split_summary["test_base_ids"]),
        len(train_dirs),
        len(val_dirs),
        len(test_dirs),
        len(train_ds),
    )
    logging.info("split files saved in: %s", out_dir / "splits")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch_i, batch in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and batch_i > int(args.max_train_batches):
                break

            x = batch["x"].to(device, non_blocking=True)
            geom = batch["geom"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            pred = model(x, geom)
            loss = sino_row_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item()) * x.size(0)
            n_train += x.size(0)

            if int(args.log_every) > 0 and batch_i % int(args.log_every) == 0:
                logging.info(
                    "epoch=%03d batch=%d seen=%d running_train_loss=%.6f",
                    epoch,
                    batch_i,
                    n_train,
                    train_loss / max(n_train, 1),
                )

        train_loss /= max(n_train, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            n_val = 0
            with torch.no_grad():
                for batch_i, batch in enumerate(val_loader, start=1):
                    if args.max_val_batches is not None and batch_i > int(args.max_val_batches):
                        break

                    x = batch["x"].to(device, non_blocking=True)
                    geom = batch["geom"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True)
                    loss = sino_row_loss(model(x, geom), y)
                    total += float(loss.item()) * x.size(0)
                    n_val += x.size(0)
            val_loss = total / max(n_val, 1)

        score = val_loss if val_loss is not None else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logging.info(
            "epoch=%03d train_loss=%.6f val_loss=%s",
            epoch,
            train_loss,
            f"{val_loss:.6f}" if val_loss is not None else "NA",
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "leaf_count": int(args.leaf_count),
            "history": history,
            "train_dirs": [str(p) for p in train_dirs],
            "val_dirs": [str(p) for p in val_dirs],
            "test_dirs": [str(p) for p in test_dirs],
            "split_summary": split_summary,
            "args": vars(args),
        }
        torch.save(checkpoint, out_dir / "latest.pt")

        if score < best_score:
            best_score = score
            torch.save(checkpoint, out_dir / "best.pt")

    write_json(out_dir / "history.json", {"history": history, "best_score": best_score})

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    eval_payload: dict[str, Any] = {
        "best_epoch": int(best_ckpt.get("epoch", -1)),
        "open_threshold": float(args.open_threshold),
        "split_summary": split_summary,
    }

    if val_dirs:
        eval_payload["val"] = evaluate_model_on_dirs(
            model=model,
            patient_dirs=val_dirs,
            device=device,
            batch_size=int(args.batch_size),
            open_threshold=float(args.open_threshold),
            split_name="val",
            out_root=out_dir if not args.no_save_eval_predictions else None,
        )

    if test_dirs:
        eval_payload["test"] = evaluate_model_on_dirs(
            model=model,
            patient_dirs=test_dirs,
            device=device,
            batch_size=int(args.batch_size),
            open_threshold=float(args.open_threshold),
            split_name="test",
            out_root=out_dir if not args.no_save_eval_predictions else None,
        )

    write_json(out_dir / "evaluation_metrics.json", eval_payload)
    logging.info("evaluation metrics saved: %s", out_dir / "evaluation_metrics.json")


def interpolate_missing_sino_rows(pred_sino: np.ndarray, cp_indices: np.ndarray) -> np.ndarray:
    n_cp, n_leaf = pred_sino.shape
    cp_indices = np.asarray(cp_indices, dtype=np.int32)

    if cp_indices.size == 0:
        return pred_sino
    if cp_indices.size == n_cp and np.all(cp_indices == np.arange(n_cp)):
        return pred_sino

    full_x = np.arange(n_cp, dtype=np.float32)
    known_x = cp_indices.astype(np.float32)
    out = pred_sino.copy()

    for leaf in range(n_leaf):
        out[:, leaf] = np.interp(full_x, known_x, pred_sino[cp_indices, leaf]).astype(np.float32)

    return out


def command_predict(args: argparse.Namespace) -> None:
    import torch

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(view_channels=2, geom_dim=5, leaf_count=int(args.leaf_count)).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.samples_file:
        patient_dirs = load_patient_dirs_from_samples_file(data_root, Path(args.samples_file))
    elif args.split and args.split != "all":
        samples_file = Path(args.checkpoint).parent / "splits" / f"{args.split}_samples.txt"
        patient_dirs = load_patient_dirs_from_samples_file(data_root, samples_file)
    else:
        patient_dirs = get_processed_patient_dirs(data_root)

    if not patient_dirs:
        raise FileNotFoundError("No patient folders selected for prediction.")

    eval_payload = evaluate_model_on_dirs(
        model=model,
        patient_dirs=patient_dirs,
        device=device,
        batch_size=int(args.batch_size),
        open_threshold=float(args.open_threshold),
        split_name=str(args.split or "selected"),
        out_root=out_root,
    )
    write_json(out_root / "prediction_metrics_summary.json", eval_payload)
    logging.info("prediction summary saved: %s", out_root / "prediction_metrics_summary.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CP-wise Tomo sinogram generation/prediction baseline")
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Inspect raw DICOM patient folders")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--out-json", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=command_inspect)

    p = sub.add_parser("preprocess", help="Create CP-wise CT+dose views and sinogram targets")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--save-views", action="store_true")
    p.add_argument("--save-volumes", action="store_true")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--view-height", type=int, default=96)
    p.add_argument("--leaf-count", type=int, default=LEAF_COUNT)
    p.add_argument("--xy-downsample", type=float, default=4.0)
    p.add_argument("--z-downsample", type=float, default=1.0)
    p.add_argument("--dose-norm-gy", type=float, default=70.0)
    p.add_argument("--cp-stride", type=int, default=1)
    p.add_argument("--max-cp", type=int, default=None)
    p.add_argument("--table-axis", choices=["z", "none"], default="z")
    p.add_argument("--ct-reducer", choices=["mean", "max", "sum"], default="mean")
    p.add_argument("--dose-reducer", choices=["mean", "max", "sum"], default="max")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=command_preprocess)

    p = sub.add_parser("train", help="Train baseline CP-view -> 64 leaves model")
    p.add_argument("--data-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--out-dir", default="./runs_cp_sino")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--open-threshold", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--leaf-count", type=int, default=LEAF_COUNT)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--no-save-eval-predictions", action="store_true")
    p.set_defaults(func=command_train)

    p = sub.add_parser("predict", help="Predict full sinograms from precomputed CP views")
    p.add_argument("--data-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-root", default="./predicted_sino")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--leaf-count", type=int, default=LEAF_COUNT)
    p.add_argument("--open-threshold", type=float, default=0.01)
    p.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    p.add_argument("--samples-file", default=None)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=command_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
