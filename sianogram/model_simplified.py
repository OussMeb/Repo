#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_simplified.py - Modern simplified training wrapper

Clean, modern approach to model training with:
- Dataclass-based configuration
- Separated concerns (metrics, visualization, checkpointing)
- Modern PyTorch practices
- Type hints throughout
"""

import logging
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend before importing pyplot

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from network import TwoStageSinoModel
from sino_metrics import SinogramMetrics, compute_sino_metrics


# ============================================================
# Configuration using dataclasses (modern Python approach)
# ============================================================

@dataclass
class DataConfig:
    """Data-related configuration."""
    path: str = '/mnt/LeGrosDisque/Julien/sianogramme/berlingo_bulked_entry_exit'
    split_json: Optional[str] = None
    ratio: float = 0.9
    random_seed: int = 44

    batch_size: int = 6
    num_workers: int = 3
    L: Optional[int] = None
    W_in: int = 64
    W: int = 64

    augment: bool = False
    cp_unit: int = 12
    cp_height: int = 12
    cp_height_px: Optional[int] = None
    split_in_two: bool = False
    patch_cp: int = 256  # legacy alias -> patch_in_cp
    patch_in_cp: Optional[int] = None
    patch_out_cp: Optional[int] = None
    halo_cp: int = 0
    jitter_cp: int = 8

    def __post_init__(self):
        cp_ref = self.cp_height if self.cp_height_px is None else self.cp_height_px
        cp_ref = int(cp_ref)
        if int(self.cp_unit) != cp_ref:
            raise ValueError(f"cp_unit ({self.cp_unit}) doit etre egal a cp_height_px ({cp_ref})")
        self.cp_unit = cp_ref
        self.cp_height = cp_ref
        self.cp_height_px = cp_ref

        out_cp = int(self.patch_out_cp) if self.patch_out_cp is not None else int(self.patch_cp)
        halo_cp = int(self.halo_cp)
        in_cp = int(self.patch_in_cp) if self.patch_in_cp is not None else int(out_cp + 2 * halo_cp)
        if in_cp < out_cp:
            raise ValueError(f"patch_in_cp ({in_cp}) doit etre >= patch_out_cp ({out_cp})")
        if in_cp != out_cp + 2 * halo_cp:
            raise ValueError(
                f"Incoherence patching: patch_in_cp ({in_cp}) != patch_out_cp ({out_cp}) + 2*halo_cp ({2 * halo_cp})"
            )
        self.patch_out_cp = out_cp
        self.patch_in_cp = in_cp
        self.halo_cp = halo_cp
        self.patch_cp = in_cp


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = 'unet'
    in_ch_drr: int = 16
    out_ch: int = 1

    base_ch: int = 64
    depth: int = 3
    num_groups: int = 8

    # FiLM conditioning
    film_extra_dim: int = 7
    use_sincos_angles: bool = True
    film_hidden: int = 512
    film_embed_dim: int = 32
    film_embed_hidden: int = 64
    use_film_all_levels: bool = True
    film_on_decoder: bool = False

    # Stem options
    stem_pre_k: int = 3
    shoulder_gate: bool = True
    shoulder_oar_index_in_oars: int = 9
    anisotropic_leafwise: bool = False
    film_light_mode: bool = False
    early_width_mode: str = 'nearest'

    # Residual blocks
    use_resblocks: bool = True
    res_dropout: float = 0.0

    # Bottleneck
    bneck_dilated_blocks: int = 1
    bneck_dropout: float = 0.05
    bneck_horiz_blocks: int = 1
    bneck_horiz_k: int = 11
    bneck_horiz_dropout: float = 0.05

    # Optional transformer
    use_transformer: bool = False
    d_model: int = 256
    nhead: int = 8
    mlp_ratio: float = 4.0
    transformer_layers: int = 4
    use_ckpt: bool = False

    # W pooling
    pool_w_levels: int = 0
    pool_w_factor: int = 2

    shape_debug: bool = False

    # Global post-stitch refiner (stage 2)
    use_global_refiner: bool = False
    refiner_hidden: int = 128
    refiner_layers: int = 4
    refiner_kernel_size: int = 5
    refiner_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    refiner_dropout: float = 0.05
    refiner_cond_dim: int = 32
    refiner_alpha_init: float = 0.0

    # Optional local CP-only head on patch outputs
    use_patch_cp_head: bool = False
    patch_cp_head_hidden: int = 128
    patch_cp_head_layers: int = 3
    patch_cp_head_kernel_size: int = 5
    patch_cp_head_dilations: list[int] = field(default_factory=lambda: [1, 2, 4])
    patch_cp_head_dropout: float = 0.05
    patch_cp_head_alpha_init: float = 0.0


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    n_epochs: int = 200
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    accum_steps: int = 1

    # AMP
    use_amp: bool = False
    amp_dtype: str = 'fp16'  # 'fp16' or 'bf16'

    # Gradient clipping
    clip_grad_norm: float = 0.5

    # Scheduler
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    scheduler_threshold: float = 1e-4
    scheduler_cooldown: int = 2

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9995

    # Checkpointing
    save_epoch_freq: int = 5
    validation_freq: int = 1
    save_best: bool = True
    best_metric: str = 'val_loss'

    # Visualization
    visual_epoch_train: int = 1
    visual_epoch_val: int = 1
    visual_batch_train: int = 50
    visual_batch_val: int = 5

    # FiLM normalization
    dose_min_gy: float = 49.0
    dose_max_gy: float = 75.0
    cp_dur_min_sec: float = 0.285
    cp_dur_max_sec: float = 0.43
    film_with_presence: bool = True

    # Metrics
    log_extra_metrics: bool = True
    wmae_alpha: float = 0.5

    # Two-stage training
    training_mode: str = 'single_stage'  # 'single_stage' | 'two_stage'
    stage1_epochs: int = 160
    stage2_epochs: int = 40
    stage2_learning_rate: float = 1e-4
    freeze_backbone_stage2: bool = True
    lambda_refiner_delta: float = 1e-3
    train_full_batch_size: int = 1


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Runtime
    device: str = 'cuda'
    expr_dir: str = 'checkpoints/default'
    resume: bool = False
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'device': self.device,
            'expr_dir': self.expr_dir,
            'resume': self.resume,
            'checkpoint_path': self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        """Create from dictionary."""
        return cls(
            data=DataConfig(**d.get('data', {})),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            device=d.get('device', 'cuda'),
            expr_dir=d.get('expr_dir', 'checkpoints/default'),
            resume=d.get('resume', False),
            checkpoint_path=d.get('checkpoint_path', None),
        )


# ============================================================
# Metrics Module
# ============================================================

class Metrics:
    """Collection of metric functions."""

    @staticmethod
    def mae_per_cp(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MAE averaged per control point."""
        p, t = y_pred.float(), y_true.float()
        return (p - t).abs().mean(dim=-1).mean()

    @staticmethod
    def pearson_corr_per_cp(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Pearson correlation per CP, robust to constant lines."""
        p = y_pred.float().squeeze(1).reshape(-1, y_pred.shape[-1])
        t = y_true.float().squeeze(1).reshape(-1, y_true.shape[-1])

        p = p - p.mean(dim=1, keepdim=True)
        t = t - t.mean(dim=1, keepdim=True)

        num = (p * t).sum(dim=1)
        den = torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1)).clamp_min(eps)

        corr = torch.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)
        return corr.mean()

    @staticmethod
    def grad_w_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MAE on horizontal gradient."""
        dp = y_pred[..., 1:] - y_pred[..., :-1]
        dt = y_true[..., 1:] - y_true[..., :-1]
        return (dp - dt).abs().mean()

    @staticmethod
    def grad_cp_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MAE on CP gradient."""
        dp = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
        dt = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        return (dp - dt).abs().mean()

    @staticmethod
    def fluence_per_cp_mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """MAE on total fluence per CP."""
        p = y_pred.float().squeeze(1).sum(dim=-1)
        t = y_true.float().squeeze(1).sum(dim=-1)
        return (p - t).abs().mean()


# ============================================================
# EMA Module
# ============================================================

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9995, device: Optional[torch.device] = None):
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.shadow = {}

        self.init_from_model(model)

    def init_from_model(self, model: nn.Module):
        """Initialize EMA from model parameters."""
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().to(device=self.device, dtype=torch.float32).clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if name not in self.shadow:
                self.shadow[name] = param.detach().to(device=self.device, dtype=torch.float32).clone()
            else:
                param_fp32 = param.detach().to(device=self.device, dtype=torch.float32)
                self.shadow[name].mul_(self.decay).add_(param_fp32, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to_model(self, model: nn.Module):
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state."""
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load EMA state."""
        self.shadow = state_dict


# ============================================================
# Visualization Module
# ============================================================

class Visualizer:
    """Handles visualization of training/validation results."""

    def __init__(self, log_dir: str, cp_height: int = 1):
        self.log_dir = Path(log_dir)
        self.cp_height = cp_height

    def visualize_patch(self, batch: Dict, y_pred: torch.Tensor, epoch: int,
                       idx: int, state: str = 'train'):
        """Visualize a training patch (first patient in batch only)."""
        import matplotlib.pyplot as plt

        save_dir = self.log_dir / f"{state}_visuals"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Extract data from first patient in batch
        patient_id = self._get_patient_id(batch)
        x_drr = batch['x_drr'][0].detach().float().cpu().numpy()  # (16, L, W_in)
        y_true = batch['y_sino'][0, 0].detach().float().cpu().numpy()
        y_pred_np = y_pred[0, 0].detach().float().cpu().numpy()

        core_start = batch.get('core_start_cp', None)
        core_end = batch.get('core_end_cp', None)
        if core_start is not None and core_end is not None:
            try:
                if torch.is_tensor(core_start):
                    cs = int(core_start.reshape(-1)[0].item())
                elif isinstance(core_start, (list, tuple)):
                    cs = int(core_start[0])
                else:
                    cs = int(core_start)

                if torch.is_tensor(core_end):
                    ce = int(core_end.reshape(-1)[0].item())
                elif isinstance(core_end, (list, tuple)):
                    ce = int(core_end[0])
                else:
                    ce = int(core_end)
                y_true = y_true[cs:ce, :]
            except Exception:
                pass

        # Compute signed error
        signed_err = np.clip(y_pred_np - y_true, -1.0, 1.0)

        # Prendre les 3 premiers canaux de X et les normaliser pour visualisation RGB
        x_rgb = x_drr[:3].transpose(1, 2, 0)  # (L, W_in, 3) - premiers 3 canaux
        # Normaliser chaque canal indépendamment pour améliorer la visualisation
        x_rgb_norm = np.zeros_like(x_rgb)
        for i in range(3):
            ch = x_rgb[:, :, i]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                x_rgb_norm[:, :, i] = (ch - ch_min) / (ch_max - ch_min)
            else:
                x_rgb_norm[:, :, i] = ch

        # Create figure with 4 panels
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        fig.suptitle(f"Patient {patient_id} - Epoch {epoch} - Iter {idx} ({state})")

        axes[0].imshow(x_rgb_norm, aspect='auto')
        axes[0].set_title('X (DRR input - channels 0,1,2)')

        axes[1].imshow(y_true, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')

        axes[2].imshow(y_pred_np, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axes[2].set_title('Prediction')

        axes[3].imshow(signed_err, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        axes[3].set_title('Signed Error')

        for ax in axes:
            ax.set_xlabel('Detector')
            ax.set_ylabel('CP index')

        plt.tight_layout()
        # Filename format: epoch_X_it_Y_patient_Z (first patient only)
        plt.savefig(save_dir / f"epoch_{epoch}_it_{idx}_patient_{patient_id}.png", dpi=100)
        plt.close(fig)

    @staticmethod
    def _get_patient_id(batch: Dict) -> str:
        """Extract patient ID from batch."""
        pid = batch.get('patient_number', 'unknown')
        if isinstance(pid, (list, tuple)) and len(pid) == 1:
            pid = pid[0]
        if hasattr(pid, 'item'):
            try:
                pid = pid.item()
            except:
                pass
        return str(pid)


# ============================================================
# Main Model Class
# ============================================================

class Model:
    """
    Modern training wrapper for sinogram prediction.

    Uses dataclass-based configuration and separated concerns.
    """

    def __init__(self, config: Config):
        self.config = config
        self.cfg_data = config.data
        self.cfg_model = config.model
        self.cfg_train = config.training

        # Setup directories
        self.expr_dir = Path(config.expr_dir)
        self.expr_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = self.expr_dir / 'TensorBoard' / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.expr_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Set random seed
        self._set_seed(self.cfg_data.random_seed)

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Build model
        self._build_model()

        # Setup training components
        self._setup_training()

        # Metrics
        # NOTE: PSNR, SSIM, Corr_CP_global removed as they don't reflect sinogram constraints
        self.metrics = Metrics()
        self.sino_metrics = SinogramMetrics(
            eps_bg=1e-6,
            t_open=1e-8,
            eps_x=1e-8,
            device=self.device
        )

        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)

        # Visualizer
        self.visualizer = Visualizer(self.log_dir, self.cfg_data.cp_height)

        # State
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Load checkpoint if resuming
        if config.resume and config.checkpoint_path:
            self.load_checkpoint(config.checkpoint_path)

        self.logger.info("Model initialized successfully")
        self._log_config()

    def _setup_logging(self):
        """Setup logging to file and console."""
        self.logger = logging.getLogger('Model')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(self.expr_dir / 'training.log')
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_model(self):
        """Build the neural network."""
        cfg = self.cfg_model
        data_cfg = self.cfg_data

        window_px = data_cfg.patch_cp * data_cfg.cp_height
        if window_px % data_cfg.cp_height != 0:
            raise ValueError(f"window_px={window_px} doit etre divisible par cp_height={data_cfg.cp_height}")

        self.model = TwoStageSinoModel.from_configs(cfg, data_cfg).to(self.device)

        self.logger.info(f"Model built: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")

    def _setup_training(self):
        """Setup optimizer, scheduler, EMA, etc."""
        cfg = self.cfg_train

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )

        # Stage-2 optimizer (global refiner only), created if refiner parameters exist.
        refiner_params = [p for p in self.model.get_refiner_parameters() if p.requires_grad]
        self.optimizer_stage2 = None
        if len(refiner_params) > 0:
            self.optimizer_stage2 = optim.Adam(
                refiner_params,
                lr=cfg.stage2_learning_rate,
                weight_decay=cfg.weight_decay,
            )
        self._stage2_prepared = False

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            threshold=cfg.scheduler_threshold,
            cooldown=cfg.scheduler_cooldown,
            min_lr=cfg.scheduler_min_lr,
            #verbose=True
        )

        # AMP scaler
        use_scaler = cfg.use_amp and cfg.amp_dtype == 'fp16'
        self.scaler = GradScaler(enabled=use_scaler)

        # EMA
        if cfg.use_ema:
            self.ema = EMA(self.model, decay=cfg.ema_decay, device=self.device)
        else:
            self.ema = None

    def _log_config(self):
        """Log configuration."""
        self.logger.info("=" * 50)
        self.logger.info("Configuration:")
        self.logger.info("=" * 50)
        config_dict = self.config.to_dict()
        for section, values in config_dict.items():
            if isinstance(values, dict):
                self.logger.info(f"\n[{section}]")
                for k, v in values.items():
                    self.logger.info(f"  {k}: {v}")
            else:
                self.logger.info(f"{section}: {values}")
        self.logger.info("=" * 50)

    def train(self, train_loader, val_loader, loss_fn, train_full_loader=None):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
        """
        cfg = self.cfg_train

        # AMP setup
        amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bf16' else torch.float16

        two_stage = (
            str(cfg.training_mode).lower() == 'two_stage'
            and bool(self.cfg_model.use_global_refiner)
        )
        stage1_epochs = int(cfg.stage1_epochs) if two_stage else int(cfg.n_epochs)
        stage2_epochs = int(cfg.stage2_epochs) if two_stage else 0
        total_epochs = int(stage1_epochs + stage2_epochs) if two_stage else int(cfg.n_epochs)

        if two_stage and train_full_loader is None:
            raise ValueError("training_mode=two_stage nécessite train_full_loader pour le stage 2")

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch

            in_stage2 = two_stage and (epoch >= stage1_epochs)
            if in_stage2:
                if not self._stage2_prepared:
                    self._prepare_stage2()
                train_metrics = self._train_epoch_refiner(train_full_loader, loss_fn, epoch, amp_dtype)
            else:
                train_metrics = self._train_epoch(train_loader, loss_fn, epoch, amp_dtype)

            # Log training metrics
            self._log_metrics(train_metrics, epoch, 'Train')

            # Validation phase
            if epoch % cfg.validation_freq == 0:
                val_metrics = self._validate_epoch(val_loader, loss_fn, epoch, amp_dtype)
                self._log_metrics(val_metrics, epoch, 'Val')

                # Scheduler only drives stage-1 optimizer.
                if not in_stage2:
                    self.scheduler.step(val_metrics['loss'])

                # Save best model
                if cfg.save_best and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"New best model! Val loss: {self.best_val_loss:.6f}")

            # Keep a single rolling latest checkpoint (overwritten each epoch)
            self.save_checkpoint(epoch, is_best=False)

        self.writer.close()
        self.logger.info("Training completed!")

    def _prepare_stage2(self):
        """Freeze backbone if requested and setup stage-2 optimizer state."""
        if not bool(self.cfg_model.use_global_refiner):
            return

        if bool(self.cfg_train.freeze_backbone_stage2):
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        for p in self.model.get_refiner_parameters():
            p.requires_grad = True

        if self.optimizer_stage2 is None:
            params = [p for p in self.model.get_refiner_parameters() if p.requires_grad]
            if len(params) == 0:
                raise RuntimeError("Aucun paramètre entraînable pour le refiner stage-2")
            self.optimizer_stage2 = optim.Adam(
                params,
                lr=self.cfg_train.stage2_learning_rate,
                weight_decay=self.cfg_train.weight_decay,
            )

        self._stage2_prepared = True
        self.logger.info(
            "Stage 2 prêt (freeze_backbone=%s, refiner_params=%d)",
            bool(self.cfg_train.freeze_backbone_stage2),
            sum(p.numel() for p in self.model.get_refiner_parameters()),
        )

    def _train_epoch(self, loader, loss_fn, epoch: int, amp_dtype) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        cfg = self.cfg_train

        metrics_accum = {
            'loss': 0.0,
            'BackgroundLeakMean': 0.0,
            'BackgroundLeakP99': 0.0,
            'NullXLeakMax': 0.0,
            'F1_open': 0.0,
            'MAE_open': 0.0,
            'MAE_EnergyCP': 0.0,
        }
        # Extra metrics for special losses (e.g., TomoSinoStrictZeroLoss)
        extra_metrics_accum = {}

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.n_epochs} [Train]", leave=False)

        for idx, batch in enumerate(pbar):
            # Move to device
            x_drr = batch['x_drr'].to(self.device, non_blocking=True)
            angles = batch['angles'].to(self.device, non_blocking=True)
            pos = batch['positions'].to(self.device, non_blocking=True)
            y_true = batch['y_sino'].to(self.device, non_blocking=True)
            film = self._prepare_film(batch['film'])

            core_start, core_end = self._extract_core_bounds(batch, cp_len=int(y_true.shape[2]))
            y_true_core = y_true[:, :, core_start:core_end, :]

            # Forward pass with AMP
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=cfg.use_amp):
                y_pred_full = self.model(x_drr, angles, pos, film)
                y_pred = y_pred_full[:, :, core_start:core_end, :]

                cp_h = int(self.cfg_data.cp_height)
                px0 = core_start * cp_h
                px1 = core_end * cp_h
                x_core = x_drr[:, :, px0:px1, :]
                loss, loss_metrics = self._compute_loss(loss_fn, y_pred, y_true_core, x_core)

            # Accumulate extra metrics
            for k, v in loss_metrics.items():
                if k not in extra_metrics_accum:
                    extra_metrics_accum[k] = 0.0
                extra_metrics_accum[k] += v

            # Check for NaN
            if not torch.isfinite(loss):
                self.logger.warning(f"NaN loss at batch {idx}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass
            loss_scaled = loss / cfg.accum_steps
            self.scaler.scale(loss_scaled).backward()

            # Optimizer step
            if (idx + 1) % cfg.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # EMA update
                if self.ema is not None:
                    self.ema.update(self.model)

            # Compute metrics
            with torch.no_grad():
                y_pred_safe = torch.clamp(torch.nan_to_num(y_pred, 0.0), 0.0, 1.0)
                metrics_accum['loss'] += loss.item()

                # Compute sinogram-specific metrics
                sino_metrics = compute_sino_metrics(
                    y_pred=y_pred_safe,
                    y_true=y_true_core,
                    x_drr=x_core,
                    denom_acc=None,
                    y_full=None,
                    include_stitching=False  # No stitching diagnostics during patch training
                )
                for k, v in sino_metrics.items():
                    if k in metrics_accum:
                        metrics_accum[k] += v

            # Update progress bar
            pbar.set_postfix(loss=f"{metrics_accum['loss']/(idx+1):.4f}")

            # Visualize
            if cfg.visual_batch_train and epoch % cfg.visual_epoch_train == 0 and idx % cfg.visual_batch_train == 0:
                self.visualizer.visualize_patch(batch, y_pred_safe, epoch, idx, 'train')

        # Average metrics
        n = len(loader)
        result = {k: v / n for k, v in metrics_accum.items()}

        # Average extra metrics
        for k, v in extra_metrics_accum.items():
            result[k] = v / n

        return result

    def _train_epoch_refiner(self, loader, loss_fn, epoch: int, amp_dtype) -> Dict[str, float]:
        """Train only the global full-sequence refiner on stitched predictions."""
        if self.optimizer_stage2 is None:
            raise RuntimeError("Stage-2 optimizer non initialisé")

        self.model.train()
        self.model.backbone.eval()
        cfg = self.cfg_train
        lambda_delta = float(cfg.lambda_refiner_delta)

        metrics_accum = {
            'loss': 0.0,
            'loss_main': 0.0,
            'loss_delta': 0.0,
            'BackgroundLeakMean_base': 0.0,
            'BackgroundLeakP99_base': 0.0,
            'NullXLeakMax_base': 0.0,
            'F1_open_base': 0.0,
            'MAE_open_base': 0.0,
            'MAE_EnergyCP_base': 0.0,
            'DenomMin_base': 0.0,
            'SeamScore_base': 0.0,
            'BackgroundLeakMean': 0.0,
            'BackgroundLeakP99': 0.0,
            'NullXLeakMax': 0.0,
            'F1_open': 0.0,
            'MAE_open': 0.0,
            'MAE_EnergyCP': 0.0,
            'DenomMin': 0.0,
            'SeamScore': 0.0,
        }

        self.optimizer_stage2.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.n_epochs} [Train-Stage2]", leave=False)

        for idx, batch in enumerate(pbar):
            x_drr = batch['x_drr'].to(self.device, non_blocking=True)
            angles = batch['angles'].to(self.device, non_blocking=True)
            pos = batch['positions'].to(self.device, non_blocking=True)
            y_true = batch['y_sino'].to(self.device, non_blocking=True)
            film = self._prepare_film(batch['film'])

            # Méthode 1 stricte: un patient complet par batch au stage 2.
            if x_drr.shape[0] != 1:
                raise RuntimeError(
                    f"Stage-2 refiner attend batch_size=1, recu {x_drr.shape[0]}. "
                    "Utiliser get_full_sequence_loader(..., batch_size=1)."
                )

            with torch.no_grad():
                y_base, denom_acc = self.predict_full_base(x_drr, angles, pos, film, amp_dtype)

            null_mask = self._compute_null_batch_mask(x_drr)
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=cfg.use_amp):
                y_refined = self.model.refine_full(
                    y_base,
                    angles=angles,
                    positions=pos,
                    film=film,
                    null_mask=null_mask,
                )
                loss_main, _ = self._compute_loss(loss_fn, y_refined, y_true, x_drr)
                loss_delta = F.l1_loss(y_refined, y_base)
                loss = loss_main + lambda_delta * loss_delta

            if not torch.isfinite(loss):
                self.logger.warning(f"NaN loss (stage2) at batch {idx}, skipping")
                self.optimizer_stage2.zero_grad(set_to_none=True)
                continue

            loss_scaled = loss / cfg.accum_steps
            self.scaler.scale(loss_scaled).backward()

            if (idx + 1) % cfg.accum_steps == 0:
                self.scaler.unscale_(self.optimizer_stage2)
                torch.nn.utils.clip_grad_norm_(self.model.get_refiner_parameters(), cfg.clip_grad_norm)
                self.scaler.step(self.optimizer_stage2)
                self.scaler.update()
                self.optimizer_stage2.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

            with torch.no_grad():
                y_base_safe = torch.clamp(torch.nan_to_num(y_base, 0.0), 0.0, 1.0)
                y_ref_safe = torch.clamp(torch.nan_to_num(y_refined, 0.0), 0.0, 1.0)

                metrics_accum['loss'] += float(loss.item())
                metrics_accum['loss_main'] += float(loss_main.item())
                metrics_accum['loss_delta'] += float(loss_delta.item())

                base_metrics = compute_sino_metrics(
                    y_pred=y_base_safe,
                    y_true=y_true,
                    x_drr=x_drr,
                    denom_acc=denom_acc,
                    y_full=y_base,
                    include_stitching=True,
                )
                refined_metrics = compute_sino_metrics(
                    y_pred=y_ref_safe,
                    y_true=y_true,
                    x_drr=x_drr,
                    denom_acc=denom_acc,
                    y_full=y_refined,
                    include_stitching=True,
                )

                for k in ['BackgroundLeakMean', 'BackgroundLeakP99', 'NullXLeakMax', 'F1_open', 'MAE_open',
                          'MAE_EnergyCP', 'DenomMin', 'SeamScore']:
                    metrics_accum[f'{k}_base'] += float(base_metrics.get(k, 0.0))
                    metrics_accum[k] += float(refined_metrics.get(k, 0.0))

            pbar.set_postfix(loss=f"{metrics_accum['loss'] / (idx + 1):.4f}")

        n = max(1, len(loader))
        return {k: v / n for k, v in metrics_accum.items()}

    @torch.no_grad()
    def _ema_swap_in(self) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        """Swap EMA weights in-place and return original params for cheap restore."""
        if self.ema is None:
            return []
        saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        for name, param in self.model.named_parameters():
            shadow = self.ema.shadow.get(name)
            if shadow is None:
                continue
            saved.append((param, param.detach().clone()))
            param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))
        return saved

    @torch.no_grad()
    def _ema_swap_out(self, saved: list[tuple[torch.nn.Parameter, torch.Tensor]]):
        for param, old in saved:
            param.data.copy_(old)

    @torch.no_grad()
    def _validate_epoch(self, loader, loss_fn, epoch: int, amp_dtype) -> Dict[str, float]:
        """Validate for one epoch."""
        ema_saved = self._ema_swap_in()

        self.model.eval()
        cfg = self.cfg_train

        metrics_accum = {
            'loss': 0.0,
            'BackgroundLeakMean': 0.0,
            'BackgroundLeakP99': 0.0,
            'NullXLeakMax': 0.0,
            'F1_open': 0.0,
            'MAE_open': 0.0,
            'MAE_EnergyCP': 0.0,
            'DenomMin': 0.0,
            'SeamScore': 0.0,
        }
        # Extra metrics for special losses
        extra_metrics_accum = {}

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.n_epochs} [Val]", leave=False)

        for idx, batch in enumerate(pbar):
            x_drr = batch['x_drr'].to(self.device, non_blocking=True)
            angles = batch['angles'].to(self.device, non_blocking=True)
            pos = batch['positions'].to(self.device, non_blocking=True)
            y_true = batch['y_sino'].to(self.device, non_blocking=True)
            film = self._prepare_film(batch['film'])

            # Predict full sequence (stage-1 stitching + optional stage-2 global refinement)
            y_pred, denom_acc, y_base = self.predict_full(
                x_drr,
                angles,
                pos,
                film,
                amp_dtype,
                return_base=True,
            )

            loss, loss_metrics = self._compute_loss(loss_fn, y_pred, y_true, x_drr)

            # Accumulate extra metrics
            for k, v in loss_metrics.items():
                if k not in extra_metrics_accum:
                    extra_metrics_accum[k] = 0.0
                extra_metrics_accum[k] += v

            # Metrics
            y_pred_safe = torch.clamp(torch.nan_to_num(y_pred, 0.0), 0.0, 1.0)
            metrics_accum['loss'] += loss.item()

            # Compute sinogram-specific metrics (with stitching diagnostics)
            sino_metrics = compute_sino_metrics(
                y_pred=y_pred_safe,
                y_true=y_true,
                x_drr=x_drr,
                denom_acc=denom_acc,
                y_full=y_pred,
                include_stitching=True  # Include stitching diagnostics during validation
            )
            for k, v in sino_metrics.items():
                if k in metrics_accum:
                    metrics_accum[k] += v

            if bool(self.cfg_model.use_global_refiner):
                y_base_safe = torch.clamp(torch.nan_to_num(y_base, 0.0), 0.0, 1.0)
                loss_base, _ = self._compute_loss(loss_fn, y_base, y_true, x_drr)
                metrics_accum['loss_base'] += loss_base.item()

                base_metrics = compute_sino_metrics(
                    y_pred=y_base_safe,
                    y_true=y_true,
                    x_drr=x_drr,
                    denom_acc=denom_acc,
                    y_full=y_base,
                    include_stitching=True,
                )
                for k in ['BackgroundLeakMean', 'BackgroundLeakP99', 'NullXLeakMax', 'F1_open', 'MAE_open',
                          'MAE_EnergyCP', 'DenomMin', 'SeamScore']:
                    metrics_accum[f'{k}_base'] += float(base_metrics.get(k, 0.0))

            pbar.set_postfix(loss=f"{metrics_accum['loss']/(idx+1):.4f}")

            # Visualize
            if cfg.visual_batch_val and epoch % cfg.visual_epoch_val == 0 and idx % cfg.visual_batch_val == 0:
                self.visualizer.visualize_patch(batch, y_pred_safe, epoch, idx, 'val')

        # Restore original weights
        self._ema_swap_out(ema_saved)

        n = len(loader)
        result = {k: v / n for k, v in metrics_accum.items()}

        # Average extra metrics
        for k, v in extra_metrics_accum.items():
            result[k] = v / n

        return result

    def _compute_loss(self, loss_fn, y_pred: torch.Tensor, y_true: torch.Tensor, x_drr: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """
        Compute loss, handling both simple losses and losses that return (loss, metrics).

        Args:
            loss_fn: Loss function
            y_pred: Predicted sinogram
            y_true: Ground truth sinogram
            x_drr: DRR input (optional, needed for TomoSinoStrictZeroLoss)

        Returns:
            (loss_scalar, metrics_dict)
        """
        try:
            # Try calling with x_drr (for losses that need it, e.g., TomoSinoStrictZeroLoss)
            result = loss_fn(y_pred, y_true, x_drr)
            if isinstance(result, tuple):
                loss, metrics = result
                return loss, metrics
            else:
                return result, {}
        except TypeError:
            # Fall back to simple call (for standard losses)
            result = loss_fn(y_pred, y_true)
            if isinstance(result, tuple):
                loss, metrics = result
                return loss, metrics
            else:
                return result, {}

    def _prepare_film(self, film) -> torch.Tensor:
        """Prepare FiLM conditioning vector."""
        if not torch.is_tensor(film):
            film = torch.as_tensor(film, dtype=torch.float32)
        film = film.to(self.device, non_blocking=True).float()
        return torch.clamp(torch.nan_to_num(film, 0.0), 0.0, 1.0)

    @staticmethod
    def _extract_core_bounds(batch: Dict[str, Any], cp_len: int) -> tuple[int, int]:
        """Read halo core bounds from batch metadata (fallback to full patch)."""
        s_v = batch.get('core_start_cp', None)
        e_v = batch.get('core_end_cp', None)

        def _to_int(v, default):
            if v is None:
                return default
            if torch.is_tensor(v):
                if v.numel() == 0:
                    return default
                return int(v.reshape(-1)[0].detach().cpu().item())
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return int(v[0])
            return int(v)

        s = _to_int(s_v, 0)
        e = _to_int(e_v, cp_len)
        s = max(0, min(s, cp_len))
        e = max(s, min(e, cp_len))
        return s, e

    def _log_metrics(self, metrics: Dict[str, float], epoch: int, phase: str):
        """Log metrics to tensorboard and console."""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{name}", value, epoch)

        use_stage2_lr = (
            str(self.cfg_train.training_mode).lower() == 'two_stage'
            and epoch >= int(self.cfg_train.stage1_epochs)
            and self.optimizer_stage2 is not None
        )
        active_opt = self.optimizer_stage2 if use_stage2_lr else self.optimizer
        lr = active_opt.param_groups[0]['lr']
        self.writer.add_scalar(f"{phase}/LR", lr, epoch)

        msg = f"[Epoch {epoch}] {phase} - " + " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        msg += f" | LR={lr:.1e}"
        self.logger.info(msg)

    @staticmethod
    def _compute_null_batch_mask(x_drr: torch.Tensor, eps_x: float = 1e-8) -> torch.Tensor:
        """Return mask [B] where sample is considered null according to hard gating rule."""
        B, C, _, _ = x_drr.shape
        if C >= 3:
            x3 = x_drr[:, :3, :, :]
            x3_norms = x3.abs().sum(dim=(1, 2, 3))
            return (x3_norms < eps_x)
        return torch.zeros((B,), device=x_drr.device, dtype=torch.bool)

    @torch.no_grad()
    def predict_full_base(self, x_drr, angles, pos, film, amp_dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage-1 full-sequence reconstruction with halo stitching.

        The model predicts on input windows of length patch_in_cp, but only the central
        patch_out_cp region is kept for reconstruction.
        """
        cfg = self.cfg_data
        patch_in_cp = int(cfg.patch_in_cp)
        patch_out_cp = int(cfg.patch_out_cp)
        halo_cp = int(cfg.halo_cp)
        overlap = patch_out_cp // 2
        step = max(1, patch_out_cp - overlap)
        cp_h = cfg.cp_height
        eps_x = 1e-8  # Threshold for detecting null patches

        B, C, H, W_in = x_drr.shape
        Ncp = H // cp_h

        # Prepare geometry
        if angles.dim() == 2:
            angles = angles.unsqueeze(-1)
        if pos.dim() == 2:
            pos = pos.unsqueeze(-1)

        # Prepare output accumulator in float32 (for numerical stability)
        recon_acc = None
        denom_acc = None
        y_pred_dtype = None

        # Generate core starts (output regions).
        if Ncp <= patch_out_cp:
            core_starts = [0]
        else:
            last_start = Ncp - patch_out_cp
            core_starts = list(range(0, last_start + 1, step))
            if not core_starts or core_starts[-1] != last_start:
                core_starts.append(last_start)

        def _extract_seq_with_edge_pad(seq: torch.Tensor, start: int, length: int) -> torch.Tensor:
            # seq: [B,Ncp,1], edge padding done by clamped index selection.
            if Ncp <= 0:
                return torch.zeros((seq.shape[0], length, seq.shape[-1]), device=seq.device, dtype=seq.dtype)
            idx = torch.arange(start, start + length, device=seq.device, dtype=torch.long)
            idx = idx.clamp(min=0, max=Ncp - 1)
            return seq.index_select(1, idx)

        null_mask = self._compute_null_batch_mask(x_drr, eps_x=eps_x)

        for core_start_cp in core_starts:
            core_end_cp = min(Ncp, core_start_cp + patch_out_cp)
            core_len = core_end_cp - core_start_cp
            in_start_cp = core_start_cp - halo_cp
            in_end_cp = in_start_cp + patch_in_cp

            # Extract input patch with zero padding on X and edge padding on geometry.
            avail_s = max(0, in_start_cp)
            avail_e = min(Ncp, in_end_cp)
            h0, h1 = avail_s * cp_h, avail_e * cp_h
            x_patch = x_drr[:, :, h0:h1, :]
            need_l = max(0, -in_start_cp)
            need_r = max(0, in_end_cp - Ncp)
            if need_l > 0 or need_r > 0:
                x_patch = F.pad(x_patch, (0, 0, need_l * cp_h, need_r * cp_h), value=0.0)

            a_patch = _extract_seq_with_edge_pad(angles, in_start_cp, patch_in_cp)
            p_patch = _extract_seq_with_edge_pad(pos, in_start_cp, patch_in_cp)

            # Forward pass
            with autocast(device_type=self.device_type, dtype=amp_dtype, enabled=self.cfg_train.use_amp):
                y_patch = self.model.predict_patch(x_patch, a_patch, p_patch, film)

            # Save dtype for final conversion
            if y_pred_dtype is None:
                y_pred_dtype = y_patch.dtype

            # Initialize accumulator on first patch
            if recon_acc is None:
                out_w = y_patch.shape[-1]
                recon_acc = torch.zeros((B, 1, Ncp, out_w), device=self.device, dtype=torch.float32)
                denom_acc = torch.zeros_like(recon_acc)

            # Keep only the core region (halo discarded from reconstruction).
            core_local_start = halo_cp
            core_local_end = core_local_start + core_len
            y_core = y_patch[:, :, core_local_start:core_local_end, :]

            if core_len > 1:
                w = torch.hann_window(core_len, periodic=False, device=self.device, dtype=torch.float32)
                w = (w / w.max()).view(1, 1, core_len, 1)
            else:
                w = torch.ones((1, 1, 1, 1), device=self.device, dtype=torch.float32)

            y_core_fp32 = y_core.float()
            recon_acc[:, :, core_start_cp:core_end_cp, :] += y_core_fp32 * w
            denom_acc[:, :, core_start_cp:core_end_cp, :] += w

        # Normalize and convert back to original dtype
        y_full = (recon_acc / denom_acc.clamp_min(1e-8)).to(dtype=y_pred_dtype)

        # ========== HARD GATING RULE: X==0 => y_pred==0 ==========
        if bool(null_mask.any()):
            y_full = y_full * (1.0 - null_mask.view(-1, 1, 1, 1).to(dtype=y_full.dtype))

        return y_full, denom_acc

    @torch.no_grad()
    def predict_full(self, x_drr, angles, pos, film, amp_dtype, return_base: bool = False):
        """
        Full pipeline prediction:
          1) patch-wise backbone + overlap-add (y_base_full)
          2) optional global CP refiner (y_final_full)
        """
        y_base_full, denom_acc = self.predict_full_base(x_drr, angles, pos, film, amp_dtype)

        if bool(self.cfg_model.use_global_refiner):
            null_mask = self._compute_null_batch_mask(x_drr)
            y_final_full = self.model.refine_full(
                y_base_full,
                angles=angles,
                positions=pos,
                film=film,
                null_mask=null_mask,
            )
        else:
            y_final_full = y_base_full

        if return_base:
            return y_final_full, denom_acc, y_base_full
        return y_final_full, denom_acc

    @staticmethod
    def _patient_id_to_str(pid: Any) -> str:
        """Preserve patient ID formatting from dataloader collation."""
        if pid is None:
            return 'unknown'
        if isinstance(pid, (list, tuple)) and len(pid) == 1:
            pid = pid[0]
        if hasattr(pid, 'item'):
            try:
                pid = pid.item()
            except Exception:
                pass
        return str(pid)

    @staticmethod
    def _scalar_from_tensor_like(value: Any) -> Optional[float]:
        """Extract one scalar float from tensor/list-like values if possible."""
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().reshape(-1)[0].cpu().item())
        if isinstance(value, (list, tuple)) and len(value) > 0:
            try:
                return float(value[0])
            except Exception:
                return None
        try:
            return float(value)
        except Exception:
            return None

    @torch.no_grad()
    def inference(
        self,
        test_loader,
        out_dir: str,
        amp: Optional[bool] = None,
        save_npz: bool = False,
        max_batches: Optional[int] = None,
        use_ema: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full-sequence inference and save one folder per patient.

        Saved outputs:
          <out_dir>/<patient_id>/y_pred.npy
          <out_dir>/<patient_id>/y_pred_base.npy (if global refiner enabled)
          <out_dir>/<patient_id>/y_pred_refined.npy (if global refiner enabled)
          <out_dir>/<patient_id>/angles_used.npy
          <out_dir>/<patient_id>/positions_used.npy
          <out_dir>/<patient_id>/film_used.npy (if available)
          <out_dir>/<patient_id>/t_used_sec.npy (if available)
        """
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        ema_saved = self._ema_swap_in() if use_ema else []
        self.model.eval()

        use_amp = bool(self.cfg_train.use_amp) if amp is None else bool(amp)
        amp_dtype = torch.bfloat16 if self.cfg_train.amp_dtype == 'bf16' else torch.float16

        n_done = 0
        patient_ids: list[str] = []

        pbar = tqdm(test_loader, desc='Inference', leave=False)
        for idx, batch in enumerate(pbar):
            if max_batches is not None and idx >= max_batches:
                break

            x_drr = batch['x_drr'].to(self.device, non_blocking=True)
            angles = batch['angles'].to(self.device, non_blocking=True)
            pos = batch['positions'].to(self.device, non_blocking=True)

            film_raw = batch.get('film', None)
            if film_raw is None:
                bsz = x_drr.shape[0]
                film = torch.zeros((bsz, int(self.cfg_model.film_extra_dim)), device=self.device, dtype=torch.float32)
            else:
                film = self._prepare_film(film_raw)

            y_pred, _, y_base = self.predict_full(
                x_drr,
                angles,
                pos,
                film,
                amp_dtype,
                return_base=True,
            )
            y_pred_safe = torch.clamp(torch.nan_to_num(y_pred, 0.0), 0.0, 1.0)

            pid = self._patient_id_to_str(batch.get('patient_number', 'unknown'))
            patient_ids.append(pid)
            p_out = out_root / pid
            p_out.mkdir(parents=True, exist_ok=True)

            y_np = y_pred_safe.detach().to(torch.float32).cpu().numpy()
            if y_np.ndim == 4 and y_np.shape[0] == 1:
                y_np = y_np.squeeze(0)

            if bool(self.cfg_model.use_global_refiner):
                y_base_np = torch.clamp(torch.nan_to_num(y_base, 0.0), 0.0, 1.0).detach().to(torch.float32).cpu().numpy()
                if y_base_np.ndim == 4 and y_base_np.shape[0] == 1:
                    y_base_np = y_base_np.squeeze(0)
                np.save(p_out / 'y_pred_base.npy', y_base_np)
                np.save(p_out / 'y_pred_refined.npy', y_np)

            np.save(p_out / 'y_pred.npy', y_np)
            np.save(p_out / 'angles_used.npy', angles.detach().to(torch.float32).cpu().numpy())
            np.save(p_out / 'positions_used.npy', pos.detach().to(torch.float32).cpu().numpy())
            np.save(p_out / 'film_used.npy', film.detach().to(torch.float32).cpu().numpy())

            t_used = self._scalar_from_tensor_like(batch.get('cp_dur_sec_mean'))
            if t_used is not None and np.isfinite(t_used) and t_used > 0:
                np.save(p_out / 't_used_sec.npy', np.array([t_used], dtype=np.float32))

            if save_npz:
                np.savez(
                    p_out / 'metadata.npz',
                    patient_id=np.array(pid),
                    y_pred_shape=np.array(y_np.shape, dtype=np.int64),
                    model=np.array(self.model.__class__.__name__),
                    device=np.array(str(self.device)),
                    use_amp=np.array(use_amp),
                    use_ema=np.array(bool(use_ema and self.ema is not None)),
                )

            n_done += 1
            pbar.set_postfix(done=n_done)

        self._ema_swap_out(ema_saved)

        report = {
            'ok': True,
            'n_patients': n_done,
            'out_dir': str(out_root),
            'use_amp': use_amp,
            'use_ema': bool(use_ema and self.ema is not None),
            'patient_ids': patient_ids,
        }
        self.logger.info(f"Inference complete: {n_done} patient(s) saved to {out_root}")
        return report

    def test(self, *args, **kwargs) -> Dict[str, Any]:
        """Backward-compatible alias to inference()."""
        return self.inference(*args, **kwargs)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint as rolling latest and optional best."""
        if not self._is_model_finite():
            self.logger.warning("Model has NaN/Inf, skipping checkpoint save")
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }

        if self.optimizer_stage2 is not None:
            checkpoint['optimizer_stage2_state_dict'] = self.optimizer_stage2.state_dict()

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        # Always overwrite latest
        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Checkpoints saved: latest={latest_path.name}, best={best_path.name}")
        else:
            self.logger.info(f"Checkpoint saved: latest={latest_path.name}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            self.logger.warning("Checkpoint missing model keys (strict=False): %s", missing)
        if unexpected:
            self.logger.warning("Checkpoint unexpected model keys (strict=False): %s", unexpected)

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.optimizer_stage2 is not None and 'optimizer_stage2_state_dict' in checkpoint:
            self.optimizer_stage2.load_state_dict(checkpoint['optimizer_stage2_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        self.logger.info(f"Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")

    def _is_model_finite(self) -> bool:
        """Check if all model parameters are finite."""
        for param in self.model.parameters():
            if not torch.isfinite(param).all():
                return False
        return True


# ============================================================
# Factory function for backward compatibility
# ============================================================

def create_model_from_legacy_dict(expr_dir: str, config_dict: Dict[str, Any], **overrides) -> Model:
    """
    Create Model from legacy dictionary configuration.

    Args:
        expr_dir: Experiment directory
        config_dict: Legacy config dictionary
        **overrides: Additional overrides

    Returns:
        Model instance
    """
    # Merge config_dict and overrides
    merged = {**config_dict, **overrides}

    # Build Config object
    config = Config(
        data=DataConfig(
            path=merged.get('path', '/mnt/LeGrosDisque/Julien/sianogramme/berlingo_bulked_entry_exit'),
            split_json=merged.get('split_json', None),
            ratio=merged.get('ratio', 0.9),
            random_seed=merged.get('random_seed', 44),
            batch_size=merged.get('batch_size', 6),
            num_workers=merged.get('num_workers', 3),
            L=merged.get('L', None),
            W_in=merged.get('W_in', 64),
            W=merged.get('W', 64),
            augment=merged.get('augment', False),
            cp_unit=merged.get('cp_unit', 12),
            cp_height=merged.get('cp_height', 12),
            split_in_two=merged.get('split_in_two', False),
            patch_cp=merged.get('patch_cp', 256),
            patch_in_cp=merged.get('patch_in_cp', None),
            patch_out_cp=merged.get('patch_out_cp', None),
            halo_cp=merged.get('halo_cp', 0),
            jitter_cp=merged.get('jitter_cp', 8),
        ),
        model=ModelConfig(
            model_name=merged.get('model_name', 'unet'),
            in_ch_drr=merged.get('in_ch_drr', 16),
            out_ch=merged.get('out_ch', 1),
            base_ch=merged.get('base_ch', 64),
            depth=merged.get('depth', 3),
            num_groups=merged.get('num_groups', 8),
            film_extra_dim=merged.get('film_extra_dim', 7),
            use_sincos_angles=merged.get('use_sincos_angles', True),
            film_hidden=merged.get('film_hidden', 512),
            film_embed_dim=merged.get('film_embed_dim', 32),
            film_embed_hidden=merged.get('film_embed_hidden', 64),
            use_film_all_levels=merged.get('use_film_all_levels', True),
            film_on_decoder=merged.get('film_on_decoder', False),
            stem_pre_k=merged.get('stem_pre_k', 3),
            shoulder_gate=merged.get('shoulder_gate', True),
            shoulder_oar_index_in_oars=merged.get('shoulder_oar_index_in_oars', 9),
            anisotropic_leafwise=merged.get('anisotropic_leafwise', False),
            film_light_mode=merged.get('film_light_mode', False),
            early_width_mode=merged.get('early_width_mode', 'nearest'),
            use_resblocks=merged.get('use_resblocks', True),
            res_dropout=merged.get('res_dropout', 0.0),
            bneck_dilated_blocks=merged.get('bneck_dilated_blocks', 1),
            bneck_dropout=merged.get('bneck_dropout', 0.05),
            bneck_horiz_blocks=merged.get('bneck_horiz_blocks', 1),
            bneck_horiz_k=merged.get('bneck_horiz_k', 11),
            bneck_horiz_dropout=merged.get('bneck_horiz_dropout', 0.05),
            use_transformer=merged.get('use_transformer', False),
            d_model=merged.get('d_model', 256),
            nhead=merged.get('nhead', 8),
            mlp_ratio=merged.get('mlp_ratio', 4.0),
            transformer_layers=merged.get('transformer_layers', 4),
            use_ckpt=merged.get('use_ckpt', False),
            pool_w_levels=merged.get('pool_w_levels', 0),
            pool_w_factor=merged.get('pool_w_factor', 2),
            shape_debug=merged.get('shape_debug', False),
            use_global_refiner=merged.get('use_global_refiner', False),
            refiner_hidden=merged.get('refiner_hidden', 128),
            refiner_layers=merged.get('refiner_layers', 4),
            refiner_kernel_size=merged.get('refiner_kernel_size', 5),
            refiner_dilations=merged.get('refiner_dilations', [1, 2, 4, 8]),
            refiner_dropout=merged.get('refiner_dropout', 0.05),
            refiner_cond_dim=merged.get('refiner_cond_dim', 32),
            refiner_alpha_init=merged.get('refiner_alpha_init', 0.0),
            use_patch_cp_head=merged.get('use_patch_cp_head', False),
            patch_cp_head_hidden=merged.get('patch_cp_head_hidden', 128),
            patch_cp_head_layers=merged.get('patch_cp_head_layers', 3),
            patch_cp_head_kernel_size=merged.get('patch_cp_head_kernel_size', 5),
            patch_cp_head_dilations=merged.get('patch_cp_head_dilations', [1, 2, 4]),
            patch_cp_head_dropout=merged.get('patch_cp_head_dropout', 0.05),
            patch_cp_head_alpha_init=merged.get('patch_cp_head_alpha_init', 0.0),
        ),
        training=TrainingConfig(
            n_epochs=merged.get('n_epochs', 200),
            learning_rate=merged.get('learning_rate', 5e-5),
            weight_decay=merged.get('weight_decay', 1e-5),
            accum_steps=merged.get('accum_steps', 1),
            use_amp=merged.get('use_amp', False),
            amp_dtype=merged.get('amp_dtype', 'fp16'),
            clip_grad_norm=merged.get('clip_grad_norm', 0.5),
            scheduler_patience=merged.get('scheduler_patience', 3),
            scheduler_factor=merged.get('scheduler_factor', 0.5),
            scheduler_min_lr=merged.get('scheduler_min_lr', 1e-7),
            scheduler_threshold=merged.get('scheduler_threshold', 1e-4),
            scheduler_cooldown=merged.get('scheduler_cooldown', 2),
            use_ema=merged.get('use_ema', True),
            ema_decay=merged.get('ema_decay', 0.9995),
            save_epoch_freq=merged.get('save_epoch_freq', 5),
            validation_freq=merged.get('validation_freq', 1),
            save_best=merged.get('save_best', True),
            best_metric=merged.get('best_metric', 'val_loss'),
            visual_epoch_train=merged.get('visual_epoch_train', 1),
            visual_epoch_val=merged.get('visual_epoch_val', 1),
            visual_batch_train=merged.get('visual_batch_train', 50),
            visual_batch_val=merged.get('visual_batch_val', 5),
            dose_min_gy=merged.get('dose_min_gy', 49.0),
            dose_max_gy=merged.get('dose_max_gy', 75.0),
            cp_dur_min_sec=merged.get('cp_dur_min_sec', 0.285),
            cp_dur_max_sec=merged.get('cp_dur_max_sec', 0.43),
            film_with_presence=merged.get('film_with_presence', True),
            log_extra_metrics=merged.get('log_extra_metrics', True),
            wmae_alpha=merged.get('wmae_alpha', 0.5),
            training_mode=merged.get('training_mode', 'single_stage'),
            stage1_epochs=merged.get('stage1_epochs', 160),
            stage2_epochs=merged.get('stage2_epochs', 40),
            stage2_learning_rate=merged.get('stage2_learning_rate', 1e-4),
            freeze_backbone_stage2=merged.get('freeze_backbone_stage2', True),
            lambda_refiner_delta=merged.get('lambda_refiner_delta', 1e-3),
            train_full_batch_size=merged.get('train_full_batch_size', 1),
        ),
        device=merged.get('device', 'cuda'),
        expr_dir=expr_dir,
        resume=merged.get('resume', False),
        checkpoint_path=merged.get('checkpoint_path', None),
    )

    return Model(config)
