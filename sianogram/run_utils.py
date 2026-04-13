#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run directory utilities for canonical training runs."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _find_git_root(start_path: str | Path) -> Path | None:
    cur = Path(start_path).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args]).decode().strip()


def save_config_used(run_dir: str | Path, config_dict: dict[str, Any]) -> None:
    run_path = Path(run_dir)
    cfg_jsonable = _jsonable(config_dict)

    with (run_path / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_jsonable, f, indent=2, sort_keys=True)

    with (run_path / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_jsonable, f, sort_keys=False, allow_unicode=False)


def save_git_state(run_dir: str | Path, repo_root: str | Path | None = None) -> tuple[str, str]:
    run_path = Path(run_dir)
    root = Path(repo_root).resolve() if repo_root else _find_git_root(run_path)

    sha = "nogit"
    diff = ""
    if root is not None:
        try:
            sha = _run_git(root, "rev-parse", "HEAD")
        except Exception:
            sha = "nogit"
        try:
            diff = _run_git(root, "diff", "--", ".")
        except Exception:
            diff = ""

    (run_path / "git_sha.txt").write_text(sha + "\n", encoding="utf-8")
    (run_path / "git_diff.patch").write_text(diff, encoding="utf-8")
    return sha, diff


def create_run_dir(run_root: str, run_name: str, config_dict: dict[str, Any]) -> str:
    run_root_path = Path(run_root).resolve()
    run_root_path.mkdir(parents=True, exist_ok=True)

    git_root = _find_git_root(Path.cwd())
    sha, _ = save_git_state(run_root_path, git_root)
    sha_short = (sha[:8] if sha and sha != "nogit" else "nogit")

    loss_name = str(config_dict.get("loss", {}).get("name", "loss")).replace(" ", "_")
    seed = config_dict.get("data", {}).get("random_seed", "na")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = str(run_name or "run").replace(" ", "_")

    run_id = f"{ts}__{safe_run_name}__loss={loss_name}__seed={seed}__sha={sha_short}"
    run_dir = run_root_path / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "TensorBoard").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    save_config_used(run_dir, config_dict)
    sha, _ = save_git_state(run_dir, git_root)

    meta = {
        "date": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git_sha": sha,
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    return str(run_dir)

