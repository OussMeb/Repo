#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical training entrypoint driven by YAML configs."""

from __future__ import annotations

import argparse
import copy
import glob
import inspect
import os
from typing import Any

import torch
import yaml

from dataloader_patches import get_full_sequence_loader, get_patch_loaders
from losses import (
    BalancedLogSpectralLoss,
    GigaUltimateLoss,
    GranularSinoLoss,
    SparseFocalSpectralLoss,
    SparseSinoLoss,
    TomoSinoStrictZeroLoss,
)
from model_simplified import Config, Model
from run_utils import create_run_dir


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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def build_loss(loss_cfg: dict[str, Any]) -> torch.nn.Module:
    loss_name = loss_cfg.get("name", "BalancedLogSpectralLoss")
    params = loss_cfg.get("params", {}) or {}
    if not isinstance(params, dict):
        raise ValueError("loss.params must be a dict")

    mapping = {
        "BalancedLogSpectralLoss": BalancedLogSpectralLoss,
        "SparseFocalSpectralLoss": SparseFocalSpectralLoss,
        "SparseSinoLoss": SparseSinoLoss,
        "GranularSinoLoss": GranularSinoLoss,
        "GigaUltimateLoss": GigaUltimateLoss,
        "TomoSinoStrictZeroLoss": TomoSinoStrictZeroLoss,
        "MSELoss": torch.nn.MSELoss,
        "L1Loss": torch.nn.L1Loss,
        "SmoothL1Loss": torch.nn.SmoothL1Loss,
    }

    if loss_name not in mapping:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(f"Unsupported loss '{loss_name}'. Supported: {supported}")

    loss_cls = mapping[loss_name]
    sig = inspect.signature(loss_cls.__init__)
    accepted = {
        p.name
        for p in sig.parameters.values()
        if p.name != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    filtered_params = {k: v for k, v in params.items() if k in accepted}
    ignored_params = sorted(set(params.keys()) - accepted)
    if ignored_params:
        print(f"[loss] Ignored unsupported params for {loss_name}: {', '.join(ignored_params)}")

    return loss_cls(**filtered_params)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train with canonical YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh run
  python train.py --config configs/base.yaml --run_name exp1

  # Resume from best checkpoint of a run
  python train.py --resume-from runs/20260304_140849__newmodel* --run_name resume_exp1

  # Resume from latest checkpoint
  python train.py --resume-from runs/20260304_140849__newmodel* --checkpoint-type latest

  # Resume with parameter overrides
  python train.py --resume-from runs/20260304_* --run_name resume_viz_fix --vis-val 1

  # Fresh run with overrides
  python train.py --config configs/base.yaml --override configs/my_exp.yaml --run_name exp1
        """,
    )

    # Config selection
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Base config YAML path (ignored if --resume-from is used). Supports glob patterns.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override YAML file(s) to merge on top of config. Supports glob. Can repeat.",
    )

    # Resume workflow
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Resume from a previous run folder. Automatically loads config_used.yaml and checkpoint. "
            "Supports glob patterns (e.g., 'runs/20260304_*__newmodel*'). "
            "Implies --run-mode resume."
        ),
    )
    parser.add_argument(
        "--checkpoint-type",
        choices=["best", "latest"],
        default="best",
        help="Which checkpoint to load when resuming (best or latest). Default: best",
    )

    # Experiment metadata
    parser.add_argument("--run_name", default="juju", help="Display name for this run")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")

    # Common parameter overrides (avoid needing separate YAML files)
    parser.add_argument(
        "--vis-val",
        type=int,
        default=None,
        help="Override validation visualization frequency (in epochs)",
    )
    parser.add_argument(
        "--vis-train",
        type=int,
        default=None,
        help="Override training visualization frequency (in epochs)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    return parser.parse_args()


def _assert_config_consistency(cfg: Config) -> None:
    cp_height_px = int(cfg.data.cp_height_px)
    if cp_height_px <= 0:
        raise ValueError("data.cp_height_px must be > 0")

    expected_hpx = int(cfg.data.patch_in_cp) * cp_height_px
    if expected_hpx <= 0:
        raise ValueError("patch_in_cp * cp_height_px must be > 0")

    if int(cfg.data.patch_in_cp) != int(cfg.data.patch_out_cp) + 2 * int(cfg.data.halo_cp):
        raise ValueError("data.patch_in_cp must equal data.patch_out_cp + 2*data.halo_cp")


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


def main() -> None:
    args = _parse_args()

    # ===== Handle resume-from workflow (high-level) =====
    if args.resume_from:
        run_dir = _find_run_dir(args.resume_from)
        config_path = os.path.join(run_dir, "config_used.yaml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config_used.yaml not found in {run_dir}")
        checkpoint_path = os.path.join(run_dir, "checkpoints", f"{args.checkpoint_type}.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"{args.checkpoint_type}.pth not found in {run_dir}/checkpoints")

        config_dict = _load_yaml(config_path)
        config_dict["resume"] = True
        config_dict["checkpoint_path"] = checkpoint_path

        print(f"[Resume] Loaded config from: {config_path}")
        print(f"[Resume] Checkpoint: {checkpoint_path}")
    else:
        # ===== Normal workflow: load from --config =====
        config_path = _resolve_single_path(args.config, flag_name="--config")
        config_dict = _load_yaml(config_path)

    # ===== Apply --override files (works for both fresh and resume) =====
    for override_path in args.override:
        resolved_override = _resolve_single_path(override_path, flag_name="--override")
        config_dict = _deep_merge(config_dict, _load_yaml(resolved_override))

    # ===== Apply CLI parameter overrides (highest priority) =====
    if args.seed is not None:
        config_dict.setdefault("data", {})
        config_dict["data"]["random_seed"] = int(args.seed)

    # Training parameter overrides
    config_dict.setdefault("training", {})
    if args.vis_train is not None:
        config_dict["training"]["visual_epoch_train"] = args.vis_train
    if args.vis_val is not None:
        config_dict["training"]["visual_epoch_val"] = args.vis_val
    if args.lr is not None:
        config_dict["training"]["learning_rate"] = args.lr
    if args.epochs is not None:
        config_dict["training"]["n_epochs"] = args.epochs

    # ===== Create run directory and finalize config =====
    run_root = config_dict.get("run_root", "runs")
    run_dir = create_run_dir(run_root=run_root, run_name=args.run_name, config_dict=config_dict)

    config_dict = copy.deepcopy(config_dict)
    config_dict["expr_dir"] = run_dir

    config = Config.from_dict(config_dict)
    _assert_config_consistency(config)

    print(f"Run directory: {run_dir}")

    train_loader, val_loader = get_patch_loaders(
        path=config.data.path,
        split_json=config.data.split_json,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        patch_cp=config.data.patch_cp,
        patch_in_cp=config.data.patch_in_cp,
        patch_out_cp=config.data.patch_out_cp,
        halo_cp=config.data.halo_cp,
        jitter_cp=config.data.jitter_cp,
        cp_height_px=config.data.cp_height_px,
        W_in=config.data.W_in,
        W_out=config.data.W,
        augment=config.data.augment,
        ratio=config.data.ratio,
        seed=config.data.random_seed,
        dose_min_gy=config.training.dose_min_gy,
        dose_max_gy=config.training.dose_max_gy,
        cp_dur_min_sec=config.training.cp_dur_min_sec,
        cp_dur_max_sec=config.training.cp_dur_max_sec,
        film_with_presence=config.training.film_with_presence,
    )

    loss_fn = build_loss(config_dict.get("loss", {}))

    train_full_loader = None
    if str(config.training.training_mode).lower() == "two_stage":
        if int(config.training.train_full_batch_size) != 1:
            raise ValueError(
                "two_stage exige training.train_full_batch_size=1 (un patient complet par batch au stage 2)."
            )
        train_full_loader = get_full_sequence_loader(
            split="train",
            path=config.data.path,
            split_json=config.data.split_json,
            batch_size=config.training.train_full_batch_size,
            num_workers=config.data.num_workers,
            patch_cp=config.data.patch_cp,
            patch_in_cp=config.data.patch_in_cp,
            patch_out_cp=config.data.patch_out_cp,
            halo_cp=config.data.halo_cp,
            cp_height_px=config.data.cp_height_px,
            W_in=config.data.W_in,
            W_out=config.data.W,
            ratio=config.data.ratio,
            seed=config.data.random_seed,
            dose_min_gy=config.training.dose_min_gy,
            dose_max_gy=config.training.dose_max_gy,
            cp_dur_min_sec=config.training.cp_dur_min_sec,
            cp_dur_max_sec=config.training.cp_dur_max_sec,
            film_with_presence=config.training.film_with_presence,
        )

    model = Model(config)
    model.train(train_loader, val_loader, loss_fn, train_full_loader=train_full_loader)


if __name__ == "__main__":
    main()
