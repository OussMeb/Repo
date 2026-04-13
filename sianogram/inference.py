#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical inference entrypoint with optional RTPLAN injection."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from dataloader_patches import get_test_loader
from model_simplified import Config, Model


def _resolve_single_path(path_value: str, *, flag_name: str) -> str:
    """Resolve a CLI path that can be a literal path or a glob pattern."""
    has_glob = glob.has_magic(path_value)
    matches = sorted(glob.glob(path_value)) if has_glob else [path_value]

    if not matches:
        raise FileNotFoundError(
            f"{flag_name}: no file matched '{path_value}'."
            " If using wildcards, ensure the pattern matches exactly one file."
        )
    if len(matches) > 1:
        joined = "\n  - ".join(matches)
        raise ValueError(
            f"{flag_name}: pattern '{path_value}' matched multiple files; "
            f"please make it unique:\n  - {joined}"
        )

    resolved = matches[0]
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{flag_name}: resolved path is not a file: {resolved}")
    return resolved


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a dict at root: {path}")
    return data


def _find_run_dir(pattern: str) -> str:
    """Find a run directory matching a glob pattern."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No run folder matched '{pattern}'")
    if len(matches) > 1:
        joined = "\n  - ".join(matches)
        raise ValueError(
            f"Pattern '{pattern}' matched multiple runs; please be more specific:\n  - {joined}"
        )
    resolved = matches[0]
    if not os.path.isdir(resolved):
        raise NotADirectoryError(f"Expected directory but got file: {resolved}")
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with canonical YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference from a previous run
  python inference.py --resume-from runs/20260304_140849__newmodel* --out-dir inference_outputs/run1

  # Inference from explicit checkpoint
  python inference.py --checkpoint runs/xxx/checkpoints/best.pth --config runs/xxx/config_used.yaml

  # Inference + direct RTPLAN injection
  python inference.py --resume-from runs/20260304_* --inject-rtplan --dicom-root /path/to/dicom_data
        """,
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Load config/checkpoint from an existing run directory. "
            "Supports glob patterns (e.g., 'runs/20260304_*')."
        ),
    )
    src_group.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit checkpoint path (.pth). Supports glob patterns.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config YAML path used with --checkpoint. "
            "If omitted, tries <checkpoint_parent_parent>/config_used.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["best", "latest"],
        default="best",
        help="Checkpoint selection when --resume-from is used.",
    )

    parser.add_argument("--out-dir", default=None, help="Output folder for predictions.")
    parser.add_argument("--data-path", default=None, help="Override data.path from config.")
    parser.add_argument(
        "--split-json",
        default=None,
        help=(
            "Override split manifest. If omitted, uses config value. "
            "If final value is empty, all subjects found in data.path are inferred."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--mmap", action="store_true", help="Use mmap for numpy loading.")
    parser.add_argument("--no-mmap", action="store_true", help="Disable mmap.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Override device.")
    parser.add_argument("--amp", action="store_true", help="Force AMP on for inference.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA weights during inference.")
    parser.add_argument("--save-npz", action="store_true", help="Save metadata.npz per patient.")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after N batches.")

    parser.add_argument(
        "--inject-rtplan",
        action="store_true",
        help="Inject predicted sinograms into DICOM RTPLAN right after inference.",
    )
    parser.add_argument(
        "--dicom-root",
        default=None,
        help="Root folder containing DICOM patient directories (required with --inject-rtplan).",
    )
    parser.add_argument(
        "--rtplan-input-mode",
        choices=["rel", "abs"],
        default="rel",
        help="Interpretation of y_pred values for injection.",
    )
    parser.add_argument(
        "--rtplan-pred-filenames",
        nargs="+",
        default=["y_pred.npy"],
        help="Prediction filenames searched per patient folder for RTPLAN injection.",
    )

    return parser.parse_args()


def _resolve_config_and_checkpoint(args: argparse.Namespace) -> tuple[str, str, dict[str, Any]]:
    if args.resume_from:
        run_dir = _find_run_dir(args.resume_from)
        config_path = os.path.join(run_dir, "config_used.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config_used.yaml not found in {run_dir}")
        checkpoint_path = os.path.join(run_dir, "checkpoints", f"{args.checkpoint_type}.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"{args.checkpoint_type}.pth not found in {run_dir}/checkpoints")
        config_dict = _load_yaml(config_path)
        return config_path, checkpoint_path, config_dict

    checkpoint_path = _resolve_single_path(args.checkpoint, flag_name="--checkpoint")
    if args.config:
        config_path = _resolve_single_path(args.config, flag_name="--config")
    else:
        run_dir_guess = os.path.dirname(os.path.dirname(checkpoint_path))
        config_path = os.path.join(run_dir_guess, "config_used.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                "Could not infer config_used.yaml from checkpoint path. "
                "Please pass --config explicitly."
            )
    config_dict = _load_yaml(config_path)
    return config_path, checkpoint_path, config_dict


def _default_out_dir(checkpoint_path: str) -> str:
    ckpt_name = Path(checkpoint_path).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("inference_outputs") / f"{stamp}__{ckpt_name}")


def main() -> None:
    args = _parse_args()
    if args.inject_rtplan and not args.dicom_root:
        raise ValueError("--dicom-root is required when --inject-rtplan is enabled")

    use_mmap = False if args.no_mmap else (True if args.mmap else True)

    config_path, checkpoint_path, config_dict = _resolve_config_and_checkpoint(args)

    config_dict = copy.deepcopy(config_dict)
    if args.data_path is not None:
        config_dict.setdefault("data", {})
        config_dict["data"]["path"] = args.data_path
    if args.split_json is not None:
        config_dict.setdefault("data", {})
        config_dict["data"]["split_json"] = args.split_json
    if args.device is not None:
        config_dict["device"] = args.device

    out_dir = args.out_dir or _default_out_dir(checkpoint_path)
    runtime_dir = str(Path(out_dir) / "_runtime")
    config_dict["expr_dir"] = runtime_dir
    config_dict["resume"] = False
    config_dict["checkpoint_path"] = None

    config = Config.from_dict(config_dict)
    model = Model(config)
    model.load_checkpoint(checkpoint_path)

    test_loader = get_test_loader(
        path=config.data.path,
        split_json=config.data.split_json,
        dose_min_gy=config.training.dose_min_gy,
        dose_max_gy=config.training.dose_max_gy,
        cp_dur_min_sec=config.training.cp_dur_min_sec,
        cp_dur_max_sec=config.training.cp_dur_max_sec,
        film_with_presence=config.training.film_with_presence,
        W_out=config.data.W,
        W_in=config.data.W_in,
        cp_height_px=config.data.cp_height_px,
        patch_cp=config.data.patch_cp,
        patch_in_cp=config.data.patch_in_cp,
        patch_out_cp=config.data.patch_out_cp,
        halo_cp=config.data.halo_cp,
        num_workers=args.num_workers,
        mmap=use_mmap,
    )

    report = model.inference(
        test_loader,
        out_dir=out_dir,
        amp=(args.amp or config.training.use_amp),
        save_npz=args.save_npz,
        max_batches=args.max_batches,
        use_ema=(not args.no_ema),
    )

    report.update(
        {
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_type": args.checkpoint_type,
            "data_path": config.data.path,
            "split_json": config.data.split_json,
        }
    )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[Inference] Patients processed: {report['n_patients']}")
    print(f"[Inference] Outputs: {out_root}")
    print(f"[Inference] Report: {report_path}")

    if args.inject_rtplan:
        from rtplan_injector import InjectorConfig, inject_from_test_root

        inj_cfg = InjectorConfig(pred_input_mode=args.rtplan_input_mode)
        inj_reports = inject_from_test_root(
            test_root=out_root,
            dicom_root=Path(args.dicom_root),
            cfg=inj_cfg,
            pred_filenames=args.rtplan_pred_filenames,
        )
        inj_path = out_root / "rtplan_injection_report.json"
        with open(inj_path, "w", encoding="utf-8") as f:
            json.dump(inj_reports, f, indent=2)

        ok = sum(1 for r in inj_reports if r.get("ok"))
        ko = len(inj_reports) - ok
        print(f"[RTPLAN] Injection done: OK={ok} KO={ko}")
        print(f"[RTPLAN] Report: {inj_path}")


if __name__ == "__main__":
    main()

