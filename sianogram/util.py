# Standard library imports
import os
import sys
import json
import shutil
import inspect
import logging
import subprocess
from datetime import datetime
from textwrap import dedent
import socket
import platform
from pathlib import Path

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.amp import autocast
from pytorch_msssim import ssim as pytorch_ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure


# Configuration
torch.backends.cudnn.benchmark = False


def prepare_experiment(expr_dir, config, run_name=None,
                       used_classes_or_modules=None,
                       files_to_copy=None,
                       resume=False, resume_epoch=None):
    """
    Crée un sous-dossier expr_dir et y stocke:
      - config.json + config.yaml
      - training.log (texte simple appendable)
      - events.out (horodaté, 1 par lancement)
      - copie des fichiers sources
    """
    os.makedirs(expr_dir, exist_ok=True)

    # 1) Sauvegarde config
    with open(os.path.join(expr_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    try:
        import yaml
        with open(os.path.join(expr_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=True, allow_unicode=True)
    except Exception:
        pass

    # 2) Copie fichiers sources (si pas déjà fait)
    files_to_copy = files_to_copy or []
    for fp in files_to_copy:
        try:
            if os.path.isfile(fp):
                dst = os.path.join(expr_dir, os.path.basename(fp))
                if not os.path.exists(dst):
                    shutil.copy2(fp, dst)
        except Exception as e:
            with open(os.path.join(expr_dir, f"copy_error_{os.path.basename(fp)}.txt"), "w") as f:
                f.write(str(e))

    # 3) Training log cumulatif
    log_path = os.path.join(expr_dir, "training.log")
    with open(log_path, "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if resume:
            f.write(f"[{now}] Resume training from epoch {resume_epoch or '?'}\n")
        else:
            f.write(f"[{now}] New training run started\n")

    # 4) Event file unique par lancement
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    event_path = os.path.join(expr_dir, f"events.{run_name}.out")
    with open(event_path, "w") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Run started at {now}\n")
        f.write(f"Resume: {resume}, epoch={resume_epoch}\n")
        f.write(json.dumps(config, indent=2, sort_keys=True))

    print(f"Experiment prepared in {expr_dir}")
    return expr_dir

def print_log(out_f, message):
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _jsonable(obj):
    """
    Transforme récursivement en types JSON-compatibles.
    Tout ce qui n'est pas primitif -> str(obj).
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _init_file_logger(log_path):
    """
    Logger dédié (n'interfère pas avec le root logger).
    """
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    # éviter doublons de handlers
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path)
               for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def snapshot_run(
    expr_dir,
    config,
    used_classes_or_modules=None,
    files_to_copy=None,
    run_name=None,
    resume=False,
    start_epoch=0,
    extra_notes=None,
):
    """
    Crée un instantané de la run dans expr_dir/snapshots/<horodatage> et enregistre:
      - config.json (+ .yaml si PyYAML dispo) avec sérialisation sûre
      - training.log (append)
      - events.out (append, 1 ligne par lancement)
      - copies des fichiers sources indiqués
      - dump du code des classes/modules utilisés
      - git_info.json (si repo)
      - requirements.txt (via pip freeze, best effort)
      - latest_config.json à la racine de expr_dir pour accès rapide

    Retourne le chemin du snapshot.
    """
    expr_dir = _ensure_dir(expr_dir)
    snaps_root = _ensure_dir(os.path.join(expr_dir, "snapshots"))

    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = _ensure_dir(os.path.join(snaps_root, run_name))

    # 1) config sérialisable
    cfg_jsonable = _jsonable(config)
    with open(os.path.join(snap_dir, "config.json"), "w") as f:
        json.dump(cfg_jsonable, f, indent=2, sort_keys=True)
    # copie "latest" à la racine de l'expérience
    with open(os.path.join(expr_dir, "latest_config.json"), "w") as f:
        json.dump(cfg_jsonable, f, indent=2, sort_keys=True)

    if _HAS_YAML:
        try:
            with open(os.path.join(snap_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg_jsonable, f, sort_keys=True, allow_unicode=True)
        except Exception as e:
            with open(os.path.join(snap_dir, "config.yaml.ERROR.txt"), "w") as f:
                f.write(str(e))

    # 2) training.log (append)
    log_path = os.path.join(expr_dir, "training.log")
    tlog = _init_file_logger(log_path)
    tlog.info("=== New run === resume=%s start_epoch=%s run_name=%s", resume, start_epoch, run_name)
    tlog.info("Config:\n%s", json.dumps(cfg_jsonable, indent=2, sort_keys=True))
    if extra_notes:
        tlog.info("Notes: %s", str(extra_notes))

    # 3) events.out (append 1 ligne)
    try:
        with open(os.path.join(expr_dir, "events.out"), "a") as f:
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"run={run_name} resume={resume} start_epoch={start_epoch} "
                f"loss={cfg_jsonable.get('loss_fn', 'n/a')} "
                f"optimizer={cfg_jsonable.get('optimizer', 'n/a')}\n"
            )
    except Exception:
        pass

    # 4) git info (best effort)
    try:
        git_info = {}
        git_info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=expr_dir).decode().strip()
        git_info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=expr_dir).decode().strip()
        with open(os.path.join(snap_dir, "git_info.json"), "w") as f:
            json.dump(git_info, f, indent=2)
    except Exception:
        # pas un repo git ou pas accessible
        pass

    # 5) requirements (best effort)
    try:
        req = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        with open(os.path.join(snap_dir, "requirements.txt"), "w") as f:
            f.write(req)
    except Exception:
        pass

    # 6) copies de fichiers sources
    for fp in files_to_copy or []:
        try:
            if os.path.isfile(fp):
                shutil.copy2(fp, os.path.join(snap_dir, os.path.basename(fp)))
        except Exception as e:
            with open(os.path.join(snap_dir, f"copy_error_{os.path.basename(fp)}.txt"), "w") as f:
                f.write(str(e))

    # 7) dump code des classes/modules
    code_dump_path = os.path.join(snap_dir, "code_snippets.py")
    dump = []
    for obj in used_classes_or_modules or []:
        try:
            src = inspect.getsource(obj)
            name = getattr(obj, "__name__", obj.__class__.__name__)
            dump.append("\n\n# ===== " + name + " =====\n" + dedent(src))
        except Exception:
            try:
                mfile = inspect.getsourcefile(obj)
                if mfile and os.path.isfile(mfile):
                    dump.append(f"\n\n# ===== FILE {mfile} =====\n")
                    with open(mfile, "r") as f:
                        dump.append(f.read())
            except Exception:
                pass
    if dump:
        with open(code_dump_path, "w") as f:
            f.write("".join(dump))

    # 8) petit README
    try:
        with open(os.path.join(snap_dir, "README.txt"), "w") as f:
            f.write(
                f"Snapshot créé le {run_name}\n"
                f"Expr dir: {expr_dir}\n"
                f"Resume: {resume}, start_epoch: {start_epoch}\n"
                f"Notes: {extra_notes or ''}\n"
            )
    except Exception:
        pass

    return snap_dir


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


def save_config_used(run_dir: str | Path, config_dict: dict):
    run_path = Path(run_dir)
    cfg_jsonable = _jsonable(config_dict)

    with (run_path / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_jsonable, f, indent=2, sort_keys=True)

    if _HAS_YAML:
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


def create_run_dir(run_root: str, run_name: str, config_dict: dict) -> str:
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


def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()
    return num_params


def resize_image(image, target_height=1024):
    """
    Redimensionner une image en l'étirant pour atteindre une hauteur cible tout en conservant la largeur.

    :param image: Tensor de taille (1, N, 64)
    :param target_height: Hauteur cible après redimensionnement (par défaut 1024)
    :return: Tensor de taille (1, target_height, 64)
    """
    # Vérifier que l'image est bien de taille (1, N, 64)
    if len(image.shape) != 3 or image.shape[0] != 1 or image.shape[2] != 64:
        raise ValueError("L'image doit être de taille (1, N, 64)")

    # Redimensionner l'image en utilisant l'interpolation bilinéaire
    resized_image = F.interpolate(image.unsqueeze(0), size=(target_height, image.shape[2]), mode='bilinear',
                                  align_corners=False)

    return resized_image.squeeze(0)


def plot_all_drrs(patient_drrs, num_angles, save_path=None, patient_id=None):
    """
    Affiche toutes les DRRs générées pour un patient avec leurs angles correspondants.

    Args:
        patient_drrs (torch.Tensor): Tensor des DRRs pour un patient de forme [num_angles, channels, H, W].
        num_angles (int): Nombre d'angles utilisés pour générer les DRRs.
        save_path (str, optional): Chemin pour sauvegarder la figure. Si None, la figure est affichée.
        patient_id (str, optional): Identifiant du patient pour l'affichage du titre.
    """
    if not isinstance(patient_drrs, torch.Tensor):
        raise TypeError("patient_drrs doit être un torch.Tensor")

    # Vérifier les dimensions et extraire les DRR
    if patient_drrs.dim() == 5:
        # [B, num_angles, channels, H, W]
        patient_drrs = patient_drrs.squeeze(0)  # [num_angles, channels, H, W]
    elif patient_drrs.dim() == 4:
        # [num_angles, channels, H, W]
        pass
    else:
        raise ValueError("patient_drrs doit avoir 4 ou 5 dimensions (batch inclus)")

    if patient_drrs.shape[1] != 1:
        raise ValueError(f"Le nombre de canaux doit être 1, mais obtenu: {patient_drrs.shape[1]}")

    # Supprimer la dimension des canaux
    patient_drrs = patient_drrs.squeeze(1)  # [num_angles, H, W]

    # Générer les angles
    angles = np.linspace(0, 360, num_angles, endpoint=False)

    # Déterminer la taille de la grille de sous-graphes
    cols = int(np.ceil(np.sqrt(num_angles)))
    rows = int(np.ceil(num_angles / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(num_angles):
        angle = angles[i]
        drr = patient_drrs[i].cpu().numpy()

        ax = axes[i]
        ax.imshow(drr, cmap='gray', aspect='auto')
        ax.set_title(f"Angle: {angle:.1f}°")
        ax.axis('off')

    # Masquer les sous-graphes vides si la grille est plus grande que le nombre d'angles
    for i in range(num_angles, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Ajouter un titre global si un identifiant de patient est fourni
    if patient_id is not None:
        fig.suptitle(f"DRRs pour le Patient {patient_id}", fontsize=16)
        plt.subplots_adjust(top=0.95)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

import os, json



# -------- helpers resume --------
def _find_expr_dir_from_checkpoint(ckpt_path: str) -> str:
    """
    Remonte depuis ckpt_path pour trouver le dossier d'expérience
    (contient latest_config.json ou training.log ou snapshots/).
    """
    d = os.path.dirname(ckpt_path)
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "latest_config.json")) \
           or os.path.isfile(os.path.join(d, "training.log")) \
           or os.path.isdir(os.path.join(d, "snapshots")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # fallback raisonnable: le parent du dossier contenant le .pth
    return os.path.dirname(os.path.dirname(ckpt_path))


def _load_latest_config(expr_dir: str) -> dict:
    cfg_path = os.path.join(expr_dir, "latest_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _apply_overrides(base_cfg: dict, override_cfg: dict) -> dict:
    """
    Applique des surcouches 'sûres' (non-architecturales) par-dessus
    la config rechargée. Tout le reste est ignoré.
    """
    SAFE_KEYS = {
        "loss_fn",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "use_amp",
        "resume",
        "checkpoint_path",
        "resume_lr_mode",
        "random_seed",
        "augment",
        "num_workers",
        "validation_freq",
        "visual_epoch_train",
        "visual_epoch_val",
        "visual_batch_train",
        "visual_batch_val",
        "scheduler_patience",
        "scheduler_factor",
        "scheduler_min_lr",
        "scheduler_threshold",
        "scheduler_cooldown",
        "jitter_cp",
        "split_in_two",
    }
    out = dict(base_cfg)
    for k, v in override_cfg.items():
        if k in SAFE_KEYS and v is not None:
            out[k] = v
    return out


def _ensure_loss_object(cfg: dict) -> None:
    """
    latest_config.json contient une version 'string' de la loss.
    On garantit que cfg['loss_fn'] est bien un objet callable.
    Si rien n’est fourni, on met L1Loss() par défaut.
    """
    loss_obj = cfg.get("loss_fn", None)
    if callable(loss_obj):
        return
    # si c'est une chaîne, on peut choisir d'ignorer et mettre L1 par défaut
    # l'utilisateur peut aussi surcharger en haut via config['loss_fn']
    cfg["loss_fn"] = torch.nn.L1Loss()

# ==== AJOUTER EN HAUT DU FICHIER, après les imports ====
import os, json

SAFE_KEYS = {
    "loss_fn", "batch_size", "learning_rate", "weight_decay", "use_amp",
    "resume", "checkpoint_path", "resume_lr_mode", "random_seed", "augment",
    "num_workers", "validation_freq", "visual_epoch_train", "visual_epoch_val",
    "visual_batch_train", "visual_batch_val", "scheduler_patience",
    "scheduler_factor", "scheduler_min_lr", "scheduler_threshold",
    "scheduler_cooldown", "jitter_cp", "split_in_two",
}

def _to_str(v):
    try:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return str(v)
        if isinstance(v, dict):
            return json.dumps({k: str(v[k]) for k in v}, ensure_ascii=False)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_to_str(x) for x in v) + "]"
        return v.__class__.__name__
    except Exception:
        return str(v)

def _compute_safe_overrides(base_cfg: dict, local_cfg: dict):
    """Liste [(key, old, new)] des surcouches SAFE réellement différentes."""
    diffs = []
    for k in SAFE_KEYS:
        if k in local_cfg and local_cfg[k] is not None:
            new = local_cfg[k]
            old = base_cfg.get(k, None)
            # on compare leur représentation string pour éviter les objets non sérialisables
            if _to_str(new) != _to_str(old):
                diffs.append((k, old, new))
    return diffs

def _prompt_resume_confirmation(expr_dir: str, ckpt_path: str, diffs):
    """
    Affiche les overrides et demande confirmation.
    Entrées acceptées: o, y, oui, yes, enter -> OK ; n, non -> annule.
    Variable d’environnement RESUME_CONFIRM=skip pour sauter la question.
    """
    if os.environ.get("RESUME_CONFIRM", "").lower() == "skip":
        print("Confirmation ignorée (RESUME_CONFIRM=skip). Reprise automatique.")
        return True

    print("\nReprise détectée.")
    print(f"Dossier d’expérience: {expr_dir}")
    print(f"Checkpoint: {ckpt_path}")

    if not diffs:
        print("Aucun paramètre modifié par rapport à latest_config.json.")
    else:
        print("Paramètres modifiés (ancienne valeur → nouvelle valeur) :")
        for k, old, new in diffs:
            print(f" • {k}: {_to_str(old)} → {_to_str(new)}")

    try:
        ans = input("OK pour reprise ? [Entrée/o=oui, n=non] ").strip().lower()
    except EOFError:
        # cas non-interactif
        print("Entrée utilisateur indisponible. Reprise par défaut.")
        return True

    if ans in ("", "o", "y", "oui", "yes"):
        return True
    return False
# ==== FIN AJOUT ====

def _find_git_root(start_path: str | Path) -> Path | None:
    """Return git root for a path, or None if path is outside a git repo."""
    cur = Path(start_path).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args]).decode().strip()


def save_config_used(run_dir: str | Path, config_dict: dict):
    """Persist resolved config for the run in JSON and YAML (when available)."""
    run_path = Path(run_dir)
    cfg_jsonable = _jsonable(config_dict)

    with (run_path / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_jsonable, f, indent=2, sort_keys=True)

    if _HAS_YAML:
        with (run_path / "config_used.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_jsonable, f, sort_keys=False, allow_unicode=False)


def save_git_state(run_dir: str | Path, repo_root: str | Path | None = None) -> tuple[str, str]:
    """Persist git SHA and current diff patch in the run directory."""
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


def create_run_dir(run_root: str, run_name: str, config_dict: dict) -> str:
    """Create canonical run folder and seed it with config/git/runtime metadata."""
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


class GigaUltimateLoss(nn.Module):
    """
    L = w_l1 * L1
        + ramp * ( w_edge_x * EdgeX + w_ffl_x * FFL_X + w_ffl2d * FFL_2D )

    EdgeX   : L1 sur le gradient horizontal (aiguise les bords en largeur).
    FFL_X   : focal frequency loss sur l’axe détecteur (FFT 1D le long de W), pondère les hautes fréquences.
    FFL_2D  : version 2D (optionnelle, à 0 par défaut ici).

    Utilisation:
        loss_fn = GigaUltimateLoss(w_l1=1.0, w_edge_x=0.05, w_ffl_x=0.10, ramp_epochs=2)
        # dans la boucle:
        # loss = loss_fn(y_pred, y_true, epoch=epoch)
    """
    def __init__(self,
                 w_l1: float = 1.0,
                 w_edge_x: float = 0.05,
                 w_ffl_x: float = 0.10,
                 w_ffl2d: float = 0.0,
                 ffl_p: float = 2.0,
                 ramp_epochs: int = 2):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_edge_x = float(w_edge_x)
        self.w_ffl_x = float(w_ffl_x)
        self.w_ffl2d = float(w_ffl2d)
        self.ffl_p = float(ffl_p)
        self.ramp_epochs = int(max(1, ramp_epochs))  # évite division par 0

    @staticmethod
    def _safe(x, v=0.0):
        return torch.nan_to_num(x, nan=v, posinf=v, neginf=v)

    @staticmethod
    def _edge_loss_x(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gp = pred[..., 1:] - pred[..., :-1]
        gt = target[..., 1:] - target[..., :-1]
        return F.l1_loss(gp, gt)

    def _ffl_x(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT 1D sur la largeur (dernier axe), pondération (f/fmax)^p
        diff = pred - target                                # [B,1,L_cp,W]
        F_diff = torch.fft.rfft(diff, dim=-1, norm='ortho') # [B,1,L_cp,W_r]
        Wr = F_diff.shape[-1]
        fx = torch.fft.rfftfreq(n=(Wr - 1) * 2, d=1.0, device=diff.device).view(1,1,1,Wr)
        w = (fx / fx.max().clamp_min(1e-9)) ** self.ffl_p
        mag = torch.view_as_real(F_diff)
        mag2 = mag[..., 0] ** 2 + mag[..., 1] ** 2
        return (w * mag2).mean()

    def _ffl_2d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Version 2D compacte (optionnelle)
        diff = pred - target
        F_diff = torch.fft.rfft2(diff, norm='ortho')        # [B,1,L_cp,W_r]
        H = diff.shape[-2]
        Wr = F_diff.shape[-1]
        fy = torch.fft.fftfreq(H, d=1.0, device=diff.device).view(1,1,H,1)
        fx = torch.fft.rfftfreq(n=(Wr - 1) * 2, d=1.0, device=diff.device).view(1,1,1,Wr)
        freq = torch.sqrt(fy**2 + fx**2)
        w = (freq / freq.max().clamp_min(1e-9)) ** self.ffl_p
        mag = torch.view_as_real(F_diff)
        mag2 = mag[..., 0] ** 2 + mag[..., 1] ** 2
        return (w * mag2).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base = self._safe(F.l1_loss(pred, target))
        if (self.w_edge_x == 0.0 and self.w_ffl_x == 0.0 and self.w_ffl2d == 0.0):
            return base

        if epoch is None:
            ramp = 1.0
        else:
            ramp = float(min(1.0, max(0.0, epoch / self.ramp_epochs)))

        loss = self.w_l1 * base

        if self.w_edge_x:
            loss = loss + ramp * self.w_edge_x * self._safe(self._edge_loss_x(pred, target))
        if self.w_ffl_x:
            loss = loss + ramp * self.w_ffl_x * self._safe(self._ffl_x(pred, target))
        if self.w_ffl2d:
            loss = loss + ramp * self.w_ffl2d * self._safe(self._ffl_2d(pred, target))

        return loss

class CombinedLoss(nn.Module):
    """
    Combinaison de L1, SSIM (via torchmetrics) et perte Fourier.
    alpha: poids L1
    beta:  poids SSIM
    gamma: poids Fourier
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta:  float = 1.0,
                 gamma: float = 0.01,
                 normalize_fourier: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.normalize_fourier = normalize_fourier

        self.l1_loss = nn.L1Loss()
        # torchmetrics SSIM, sans device hard-codé
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Déplacer SSIM sur le device des entrées
        self.ssim = self.ssim.to(pred.device)

        # 1) L1
        l1 = self.l1_loss(pred, target)

        # 2) SSIM
        ssim_val = self.ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        # 3) Fourier loss
        pred_f   = torch.fft.rfft2(pred.float(),   norm='ortho')
        target_f = torch.fft.rfft2(target.float(), norm='ortho')
        mag_p    = torch.abs(pred_f)
        mag_t    = torch.abs(target_f)
        fourier_loss = F.mse_loss(mag_p, mag_t)
        if self.normalize_fourier:
            fourier_loss = fourier_loss / mag_p.numel()

        # 4) somme pondérée
        total = (self.alpha * l1
               + self.beta  * ssim_loss
               + self.gamma * fourier_loss)
        return total
class UltimateLoss(nn.Module):
    """
    Loss multi-�chelle :
     - scale 1.0 (full)
     - scale 0.5
     - scale 0.25
    Chaque échelle pèse gamma_i.
    """
    def __init__(self,
                 alpha=1.0, beta=1.0, gamma=0.01,
                 weights=(0.6, 0.3, 0.1),
                 normalize_fourier=True):
        super().__init__()
        self.weights = weights
        # instantiate one CombinedLoss (they partagent les mêmes coeffs)
        self.base_loss = CombinedLoss(alpha=alpha,
                                      beta=beta,
                                      gamma=gamma,
                                      normalize_fourier=normalize_fourier)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B,1,H,W] ou [B,1,D,H,W] (on écrase la dernière dim si besoin).
        """
        # 1) assure qu'on est en 4D [B,1,H,W]
        if pred.dim() == 5:
            pred = pred.mean(dim=-1)
            target = target.mean(dim=-1)

        total_loss = 0.0
        B, C, H, W = pred.shape

        # échelles et pondérations
        scales = [1.0, 0.5, 0.25]
        for scale, w in zip(scales, self.weights):
            if w == 0:
                continue
            # redimensionne pred & target à la taille échelle
            if scale != 1.0:
                size = (int(H * scale), int(W * scale))
                p = F.interpolate(pred, size=size, mode='bilinear', align_corners=False)
                t = F.interpolate(target, size=size, mode='bilinear', align_corners=False)
            else:
                p, t = pred, target

            # calcule la loss
            l = self.base_loss(p, t)
            total_loss = total_loss + w * l

        return total_loss
def visualize_data(batch_data):
    """
    Visualise les sinogrammes, les images des organes et la somme des PTV.

    :param batch_data: Un dictionnaire contenant 'x', 'sino', 'ptv_sum', et 'patient_number'.
    """
    x = batch_data['x']          # [batch_size, num_organs, 64, 1024]
    sino = batch_data['sino']    # [batch_size, 1024, 64]
    ptv_sum = batch_data['ptv_sum']  # [batch_size, 1, 64, 1024]
    patient_number = batch_data['patient_number']

    # Convertir les tensors en numpy arrays
    x_np = x.squeeze(0).cpu().numpy()            # [num_organs, 64, 1024]
    sino_np = sino.squeeze(0).cpu().numpy()      # [1024, 64]
    ptv_sum_np = ptv_sum.squeeze(0).cpu().numpy()  # [1, 64, 1024]

    # **1. Visualiser le Sinogramme**
    plt.figure(figsize=(10, 4))
    plt.imshow(sino_np, aspect='auto', cmap='gray')
    plt.title(f'Patient {patient_number} - Sinogramme')
    plt.xlabel('Détecteur')
    plt.ylabel('Angle de projection')
    plt.colorbar()
    plt.show()

    # **2. Visualiser la Somme des PTV**
    plt.figure(figsize=(10, 4))
    plt.imshow(ptv_sum_np[0], aspect='auto', cmap='hot')
    plt.title(f'Patient {patient_number} - Somme des PTV')
    plt.xlabel('Largeur')
    plt.ylabel('Hauteur')
    plt.colorbar()
    plt.show()

    # **3. Visualiser les Images des Organes**
    num_organs = x_np.shape[0]
    num_organs_to_plot = min(5, num_organs)  # Limiter à 5 organes pour la clarté

    fig, axs = plt.subplots(1, num_organs_to_plot, figsize=(20, 4))
    for i in range(num_organs_to_plot):
        axs[i].imshow(x_np[i], aspect='auto', cmap='gray')
        axs[i].set_title(f'Organe {i+1}')
        axs[i].axis('off')
    plt.suptitle(f'Patient {patient_number} - Images des Organes')
    plt.show()


def visualize_all_organs(batch_data):
    """
    Visualise tous les organes pour un batch de données.

    :param batch_data: Un dictionnaire contenant 'x', 'sino', 'ptv_sum', et 'patient_number'.
    """
    x = batch_data['x']          # [batch_size, num_organs, 64, 1024]
    sino = batch_data['sino']    # [batch_size, 1024, 64]
    ptv_sum = batch_data['ptv_sum']  # [batch_size, 1, 64, 1024]
    patient_number = batch_data['patient_number']

    # Convertir les tensors en numpy arrays
    x_np = x.squeeze(0).cpu().numpy()            # [num_organs, 64, 1024]
    sino_np = sino.squeeze(0).cpu().numpy()      # [1024, 64]
    ptv_sum_np = ptv_sum.squeeze(0).cpu().numpy()  # [1, 64, 1024]

    num_organs = x_np.shape[0]
    cols = 5
    rows = int(np.ceil(num_organs / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axs = axs.flatten()

    for i in range(num_organs):
        axs[i].imshow(x_np[i], aspect='auto', cmap='gray')
        axs[i].set_title(f'Organe {i+1}')
        axs[i].axis('off')

    # Cacher les sous-plots inutilisés
    for j in range(num_organs, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(f'Patient {patient_number} - Tous les Organes')
    plt.tight_layout()
    plt.show()

def trier_elements(elements):
    """
    Trie une liste d'éléments en mettant les PTV en premier (par ordre décroissant de leur numéro)
    puis les organes (par ordre alphabétique).

    Args:
        elements (set): Ensemble ou liste d'éléments à trier.

    Returns:
        list: Liste triée avec les PTV en premier et les organes ensuite.
    """
    ptv = []
    organes = []

    for item in elements:
        if item.startswith('PTV_'):
            try:
                # Extraire le numéro après 'PTV_'
                numero = int(item.split('_')[1])
                ptv.append((numero, item))
            except (IndexError, ValueError):
                # Ajouter à organes si le format n'est pas correct
                organes.append(item)
        else:
            organes.append(item)

    # Trier les PTV par numéro décroissant
    ptv_sorted = sorted(ptv, key=lambda x: x[0], reverse=True)
    # Extraire uniquement les noms des PTV triés
    ptv_sorted_names = [item[1] for item in ptv_sorted]

    # Trier les organes par ordre alphabétique
    organes_sorted = sorted(organes)

    # Combiner les PTV triés et les organes triés
    return ptv_sorted_names + organes_sorted


# Fonctions utilitaires pour la génération des DRR
class MAESSIMLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, data_range=1.0):
        """
        Combinaison de MAE et SSIM pour la loss.

        Args:
            alpha (float): Poids pour la MAE.
            beta (float): Poids pour le SSIM.
            data_range (float): Plage des données, par défaut 1.0 (si les données sont normalisées entre 0 et 1).
        """
        super(MAESSIMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mae = nn.MSELoss()
        self.data_range = data_range

    def forward(self, pred, target):
        """
        Calcul de la loss combinée.

        Args:
            pred (Tensor): Prédictions du modèle [T, B, feature_size].
            target (Tensor): Cibles réelles [T, B, feature_size].

        Returns:
            Tensor: Valeur de la loss combinée.
        """
        # Calcul de la MAE
        mae_loss = self.mae(pred, target)

        # Reshape pour SSIM
        # SSIM attend des images en [B, C, H, W]. Supposons que feature_size = W et T = H
        pred_ssim = pred.permute(1, 2, 0).unsqueeze(1)  # [B, 1, W, T]
        target_ssim = target.permute(1, 2, 0).unsqueeze(1)  # [B, 1, W, T]

        # Calcul du SSIM
        ssim_loss = 1 - pytorch_ssim(pred_ssim, target_ssim, data_range=self.data_range, size_average=True)

        # Combinaison des pertes
        loss = self.alpha * mae_loss + self.beta * ssim_loss
        return loss


def hu_to_mu(volume):
    # Convert Hounsfield units to linear attenuation coefficients
    mu_water = 0.02  # Attenuation coefficient of water
    mu = mu_water * (1 + volume / 1000.0)
    mu = torch.clamp(mu, min=0)  # Ensure μ is positive
    return mu

def rotate_volume(volume, angle):
    """
    Rotate the volume around the Z-axis by the given angle.

    Args:
        volume: Tensor of shape [batch_size, channels, D, H, W]
        angle: Rotation angle in degrees

    Returns:
        Rotated volume: Tensor of shape [batch_size, channels, D, H, W]
    """
    batch_size, channels, D, H, W = volume.size()
    device = volume.device
    angle_rad = np.deg2rad(angle)

    # Create rotation matrix around Z-axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta,  cos_theta, 0, 0],
        [0,          0,         1, 0]
    ], dtype=torch.float32, device=device)  # [3, 4]

    # Expand rotation matrix to [batch_size, 3, 4]
    rotation_matrix = rotation_matrix.unsqueeze(0).expand(batch_size, -1, -1)

    # Create affine grid
    grid = F.affine_grid(rotation_matrix, size=volume.size(), align_corners=False)

    # Apply grid sample
    rotated_volume = F.grid_sample(volume, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    return rotated_volume

def generate_drr(volume):
    """
    Génère un DRR en projetant le volume le long de l'axe W (gauche-droite).

    Args:
        volume: Tensor de forme [batch_size, channels, D, H, W]

    Returns:
        drr: Tensor de forme [batch_size, channels, H, D]
    """
    # Convertir le volume de HU à μ
    mu = hu_to_mu(volume)
    # Calculer l'intégrale linéique le long de l'axe W
    line_integral = torch.sum(mu, dim=4)  # Projection sur l'axe W (dimension 4)
    # Simuler l'atténuation exponentielle
    drr = torch.exp(-line_integral)
    # Normaliser le DRR
    drr = (drr - drr.min()) / (drr.max() - drr.min() + 1e-8)
    # Permuter les dimensions pour obtenir [batch_size, channels, H, D]
    drr = drr.permute(0, 1, 3, 2)  # Échanger H et D pour avoir la taille [H, D]
    return drr  # [batch_size, channels, H, D]

def generate_drrs(volume, num_angles=51):
    """
    Génère une série de DRR en tournant le volume autour de l'axe Z.

    Args:
        volume: Tensor de forme [batch_size, channels, D, H, W]
        num_angles: Nombre de projections à générer

    Returns:
        drrs: Tensor de forme [batch_size, num_angles, channels, H, D]
    """
    batch_size = volume.size(0)
    device = volume.device
    angles = np.linspace(0, 360, num_angles, endpoint=False)

    drrs_batch = []

    for b in range(batch_size):
        vol = volume[b:b+1]  # [1, channels, D, H, W]

        drrs_sample = []

        for angle in angles:
            # Tourner le volume autour de l'axe Z
            rotated_vol = rotate_volume(vol, angle)  # [1, channels, D, H, W]

            # Générer le DRR en projetant le long de l'axe W
            drr = generate_drr(rotated_vol)  # [1, channels, H, D]

            drrs_sample.append(drr[0])  # [channels, H, D]

        # Empiler les DRR pour cet échantillon
        drrs_sample = torch.stack(drrs_sample, dim=0)  # [num_angles, channels, H, D]
        drrs_batch.append(drrs_sample)

    # Empiler les DRR pour tout le batch
    drrs = torch.stack(drrs_batch, dim=0)  # [batch_size, num_angles, channels, H, D]
    return drrs.to(device)

def normalize_to_zero_to_one(x):
    """
    Normalise un tenseur pour que ses valeurs soient entre 0 et 1.

    Args:
        x (torch.Tensor): Tenseur d'entrée.

    Returns:
        torch.Tensor: Tenseur normalisé entre 0 et 1.
    """
    return (x - x.min()) / (x.max() - x.min())

def unnormalize(x):
    """
    Fonction d'unnormalisation. Dans ce contexte, elle est une identité car les données sont déjà entre 0 et 1.

    Args:
        x (torch.Tensor): Tenseur normalisé.

    Returns:
        torch.Tensor: Tenseur non modifié.
    """
    return x


class FourierLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, data_range=1.0, win_size=7, visualize=True):
        """
        Args:
            alpha (float): Poids pour la perte spatiale (MSE).
            beta (float): Poids pour la perte Fourier (MSE sur les parties réelles et imaginaires).
            data_range (float): Plage des données (par exemple, 1.0 pour [0,1]).
            win_size (int): Taille de la fenêtre pour toute transformation ou filtre si nécessaire.
            visualize (bool): Indique si les visualisations doivent être affichées pour le débogage.
        """
        super(FourierLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.data_range = data_range
        self.win_size = win_size
        self.visualize = visualize
        self.mse_loss = nn.MSELoss()
        self.freq = 0.1


    def forward(self, predicted, target):
        """
        Args:
            predicted (Tensor): Images prédites avec forme [B, C, H, W].
            target (Tensor): Images cibles avec forme [B, C, H, W].
        Returns:
            Tensor: Perte totale combinée.
        """
        # Perte dans le domaine spatial
        spatial_loss = self.mse_loss(predicted, target)

        # Transformation de Fourier 2D
        predicted_fft = torch.fft.fft2(predicted, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        # Séparation des parties réelles et imaginaires
        predicted_real = predicted_fft.real
        predicted_imag = predicted_fft.imag
        target_real = target_fft.real
        target_imag = target_fft.imag

        # Perte MSE sur les parties réelles et imaginaires
        fft_real_loss = self.mse_loss(predicted_real, target_real)
        fft_imag_loss = self.mse_loss(predicted_imag, target_imag)

        # Perte totale dans le domaine Fourier
        fourier_loss = fft_real_loss + fft_imag_loss

        # Perte combinée
        total_loss = self.alpha * spatial_loss + self.beta * fourier_loss

        # Visualisation optionnelle pour le débogage
        if self.visualize and torch.rand(1).item() < self.freq:  # Visualiser environ 1% des batches
            self.visualize_predictions(predicted, target, predicted_fft, target_fft)

        return total_loss

    def visualize_predictions(self, predicted, target, predicted_fft, target_fft):
        """
        Visualise des prédictions et leurs transformations de Fourier pour le débogage.
        """
        # Sélectionner un échantillon aléatoire du batch
        idx = torch.randint(0, predicted.size(0), (1,)).item()

        pred_img = predicted[idx].detach().cpu().numpy()
        target_img = target[idx].detach().cpu().numpy()

        pred_fft_real = predicted_fft.real[idx].detach().cpu().numpy()
        pred_fft_imag = predicted_fft.imag[idx].detach().cpu().numpy()
        target_fft_real = target_fft.real[idx].detach().cpu().numpy()
        target_fft_imag = target_fft.imag[idx].detach().cpu().numpy()

        # Calculer les magnitudes des spectres
        pred_fft_mag = np.abs(pred_fft_real + 1j * pred_fft_imag)
        target_fft_mag = np.abs(target_fft_real + 1j * target_fft_imag)

        # Centrer les spectres de Fourier
        pred_fft_mag_shifted = np.fft.fftshift(pred_fft_mag)
        target_fft_mag_shifted = np.fft.fftshift(target_fft_mag)

        # Tracer les images et leurs spectres de Fourier
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Image prédit vs cible
        axes[0, 0].imshow(pred_img.squeeze(), cmap='gray',aspect='auto')
        axes[0, 0].set_title('Image Prédite')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(target_img.squeeze(), cmap='gray',aspect='auto')
        axes[0, 1].set_title('Image Cible')
        axes[0, 1].axis('off')

        # Spectre de Fourier prédit vs cible
        axes[1, 0].imshow(pred_fft_mag_shifted, cmap='viridis',aspect='auto')
        axes[1, 0].set_title('Spectre Fourier Prédit')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(target_fft_mag_shifted, cmap='viridis',aspect='auto')
        axes[1, 1].set_title('Spectre Fourier Cible')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()



def print_network(net, out_f=None):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    if out_f is not None:
        out_f.write(net.__repr__() + "\n")
        out_f.write('Total number of parameters: %d\n' % num_params)
        out_f.flush()
    return num_params


def resize_image(image, target_height=1024):
    """
    Redimensionner une image en l'étirant pour atteindre une hauteur cible tout en conservant la largeur.

    :param image: Tensor de taille (1, N, 64)
    :param target_height: Hauteur cible après redimensionnement (par défaut 1024)
    :return: Tensor de taille (1, target_height, 64)
    """
    # Vérifier que l'image est bien de taille (1, N, 64)
    if len(image.shape) != 3 or image.shape[0] != 1 or image.shape[2] != 64:
        raise ValueError("L'image doit être de taille (1, N, 64)")

    # Redimensionner l'image en utilisant l'interpolation bilinéaire
    resized_image = F.interpolate(image.unsqueeze(0), size=(target_height, image.shape[2]), mode='bilinear',
                                  align_corners=False)

    return resized_image.squeeze(0)


def plot_all_drrs(patient_drrs, num_angles, save_path=None, patient_id=None):
    """
    Affiche toutes les DRRs générées pour un patient avec leurs angles correspondants.

    Args:
        patient_drrs (torch.Tensor): Tensor des DRRs pour un patient de forme [num_angles, channels, H, W].
        num_angles (int): Nombre d'angles utilisés pour générer les DRRs.
        save_path (str, optional): Chemin pour sauvegarder la figure. Si None, la figure est affichée.
        patient_id (str, optional): Identifiant du patient pour l'affichage du titre.
    """
    if not isinstance(patient_drrs, torch.Tensor):
        raise TypeError("patient_drrs doit être un torch.Tensor")

    # Vérifier les dimensions et extraire les DRR
    if patient_drrs.dim() == 5:
        # [B, num_angles, channels, H, W]
        patient_drrs = patient_drrs.squeeze(0)  # [num_angles, channels, H, W]
    elif patient_drrs.dim() == 4:
        # [num_angles, channels, H, W]
        pass
    else:
        raise ValueError("patient_drrs doit avoir 4 ou 5 dimensions (batch inclus)")

    if patient_drrs.shape[1] != 1:
        raise ValueError(f"Le nombre de canaux doit être 1, mais obtenu: {patient_drrs.shape[1]}")

    # Supprimer la dimension des canaux
    patient_drrs = patient_drrs.squeeze(1)  # [num_angles, H, W]

    # Générer les angles
    angles = np.linspace(0, 360, num_angles, endpoint=False)

    # Déterminer la taille de la grille de sous-graphes
    cols = int(np.ceil(np.sqrt(num_angles)))
    rows = int(np.ceil(num_angles / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(num_angles):
        angle = angles[i]
        drr = patient_drrs[i].cpu().numpy()

        ax = axes[i]
        ax.imshow(drr, cmap='gray', aspect='auto')
        ax.set_title(f"Angle: {angle:.1f}°")
        ax.axis('off')

    # Masquer les sous-graphes vides si la grille est plus grande que le nombre d'angles
    for i in range(num_angles, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Ajouter un titre global si un identifiant de patient est fourni
    if patient_id is not None:
        fig.suptitle(f"DRRs pour le Patient {patient_id}", fontsize=16)
        plt.subplots_adjust(top=0.95)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

import os, json



# -------- helpers resume --------
def _find_expr_dir_from_checkpoint(ckpt_path: str) -> str:
    """
    Remonte depuis ckpt_path pour trouver le dossier d'expérience
    (contient latest_config.json ou training.log ou snapshots/).
    """
    d = os.path.dirname(ckpt_path)
    for _ in range(5):
        if os.path.isfile(os.path.join(d, "latest_config.json")) \
           or os.path.isfile(os.path.join(d, "training.log")) \
           or os.path.isdir(os.path.join(d, "snapshots")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # fallback raisonnable: le parent du dossier contenant le .pth
    return os.path.dirname(os.path.dirname(ckpt_path))


def _load_latest_config(expr_dir: str) -> dict:
    cfg_path = os.path.join(expr_dir, "latest_config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)


def _apply_overrides(base_cfg: dict, override_cfg: dict) -> dict:
    """
    Applique des surcouches 'sûres' (non-architecturales) par-dessus
    la config rechargée. Tout le reste est ignoré.
    """
    SAFE_KEYS = {
        "loss_fn",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "use_amp",
        "resume",
        "checkpoint_path",
        "resume_lr_mode",
        "random_seed",
        "augment",
        "num_workers",
        "validation_freq",
        "visual_epoch_train",
        "visual_epoch_val",
        "visual_batch_train",
        "visual_batch_val",
        "scheduler_patience",
        "scheduler_factor",
        "scheduler_min_lr",
        "scheduler_threshold",
        "scheduler_cooldown",
        "jitter_cp",
        "split_in_two",
    }
    out = dict(base_cfg)
    for k, v in override_cfg.items():
        if k in SAFE_KEYS and v is not None:
            out[k] = v
    return out


def _ensure_loss_object(cfg: dict) -> None:
    """
    latest_config.json contient une version 'string' de la loss.
    On garantit que cfg['loss_fn'] est bien un objet callable.
    Si rien n’est fourni, on met L1Loss() par défaut.
    """
    loss_obj = cfg.get("loss_fn", None)
    if callable(loss_obj):
        return
    # si c'est une chaîne, on peut choisir d'ignorer et mettre L1 par défaut
    # l'utilisateur peut aussi surcharger en haut via config['loss_fn']
    cfg["loss_fn"] = torch.nn.L1Loss()

# ==== AJOUTER EN HAUT DU FICHIER, après les imports ====
import os, json

SAFE_KEYS = {
    "loss_fn", "batch_size", "learning_rate", "weight_decay", "use_amp",
    "resume", "checkpoint_path", "resume_lr_mode", "random_seed", "augment",
    "num_workers", "validation_freq", "visual_epoch_train", "visual_epoch_val",
    "visual_batch_train", "visual_batch_val", "scheduler_patience",
    "scheduler_factor", "scheduler_min_lr", "scheduler_threshold",
    "scheduler_cooldown", "jitter_cp", "split_in_two",
}

def _to_str(v):
    try:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return str(v)
        if isinstance(v, dict):
            return json.dumps({k: str(v[k]) for k in v}, ensure_ascii=False)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_to_str(x) for x in v) + "]"
        return v.__class__.__name__
    except Exception:
        return str(v)

def _compute_safe_overrides(base_cfg: dict, local_cfg: dict):
    """Liste [(key, old, new)] des surcouches SAFE réellement différentes."""
    diffs = []
    for k in SAFE_KEYS:
        if k in local_cfg and local_cfg[k] is not None:
            new = local_cfg[k]
            old = base_cfg.get(k, None)
            # on compare leur représentation string pour éviter les objets non sérialisables
            if _to_str(new) != _to_str(old):
                diffs.append((k, old, new))
    return diffs

def _prompt_resume_confirmation(expr_dir: str, ckpt_path: str, diffs):
    """
    Affiche les overrides et demande confirmation.
    Entrées acceptées: o, y, oui, yes, enter -> OK ; n, non -> annule.
    Variable d’environnement RESUME_CONFIRM=skip pour sauter la question.
    """
    if os.environ.get("RESUME_CONFIRM", "").lower() == "skip":
        print("Confirmation ignorée (RESUME_CONFIRM=skip). Reprise automatique.")
        return True

    print("\nReprise détectée.")
    print(f"Dossier d’expérience: {expr_dir}")
    print(f"Checkpoint: {ckpt_path}")

    if not diffs:
        print("Aucun paramètre modifié par rapport à latest_config.json.")
    else:
        print("Paramètres modifiés (ancienne valeur → nouvelle valeur) :")
        for k, old, new in diffs:
            print(f" • {k}: {_to_str(old)} → {_to_str(new)}")

    try:
        ans = input("OK pour reprise ? [Entrée/o=oui, n=non] ").strip().lower()
    except EOFError:
        # cas non-interactif
        print("Entrée utilisateur indisponible. Reprise par défaut.")
        return True

    if ans in ("", "o", "y", "oui", "yes"):
        return True
    return False
# ==== FIN AJOUT ====

def _find_git_root(start_path: str | Path) -> Path | None:
    """Return git root for a path, or None if path is outside a git repo."""
    cur = Path(start_path).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _run_git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args]).decode().strip()


def save_config_used(run_dir: str | Path, config_dict: dict):
    """Persist resolved config for the run in JSON and YAML (when available)."""
    run_path = Path(run_dir)
    cfg_jsonable = _jsonable(config_dict)

    with (run_path / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_jsonable, f, indent=2, sort_keys=True)

    if _HAS_YAML:
        with (run_path / "config_used.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_jsonable, f, sort_keys=False, allow_unicode=False)


def save_git_state(run_dir: str | Path, repo_root: str | Path | None = None) -> tuple[str, str]:
    """Persist git SHA and current diff patch in the run directory."""
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


def create_run_dir(run_root: str, run_name: str, config_dict: dict) -> str:
    """Create canonical run folder and seed it with config/git/runtime metadata."""
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
