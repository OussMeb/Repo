#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sino_metrics.py - Specialized metrics for sinogram reconstruction

Implements metrics aligned with sinogram constraints:
- Background leakage detection
- Null X rule enforcement (X==0 => Y==0)
- Occupancy (punch card) metrics
- Amplitude errors on open regions
- Energy per control point
- Stitching diagnostics
"""

from typing import Dict, Optional, Tuple
import torch
import numpy as np


class SinogramMetrics:
    """Compute specialized metrics for sinogram reconstruction."""

    def __init__(
        self,
        eps_bg: float = 1e-6,
        t_open: float = 1e-8,
        eps_x: float = 1e-8,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            eps_bg: Threshold for background detection in y_true
            t_open: Threshold for openings detection in y_pred
            eps_x: Threshold for null X detection
            device: Device to use for computations
        """
        self.eps_bg = float(eps_bg)
        self.t_open = float(t_open)
        self.eps_x = float(eps_x)
        self.device = device

    @staticmethod
    def _quantile_fp(tensor: torch.Tensor, q: float) -> torch.Tensor:
        """torch.quantile requires float/double; AMP can produce bf16/half tensors."""
        if tensor.dtype not in (torch.float32, torch.float64):
            tensor = tensor.float()
        return torch.quantile(tensor, q)

    def compute_background_leakage(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute background leakage metrics.

        mask_bg = (abs(y_true) <= eps_bg)
        BackgroundLeakMean = mean(abs(y_pred) on mask_bg)
        BackgroundLeakP99 = 99.9 percentile of abs(y_pred) on mask_bg
        BackgroundLeakMax = max(abs(y_pred) on mask_bg)

        Args:
            y_pred: Predicted sinogram [B, 1, CP, W]
            y_true: Ground truth sinogram [B, 1, CP, W]

        Returns:
            Dict with keys: BackgroundLeakMean, BackgroundLeakP99, BackgroundLeakMax
        """
        # Create background mask
        mask_bg = torch.abs(y_true) <= self.eps_bg

        # Extract background predictions
        bg_pred = torch.abs(y_pred[mask_bg])

        if len(bg_pred) == 0:
            return {
                'BackgroundLeakMean': 0.0,
                'BackgroundLeakP99': 0.0,
                'BackgroundLeakMax': 0.0,
            }

        mean_leak = bg_pred.mean().item()
        max_leak = bg_pred.max().item()
        p99_leak = self._quantile_fp(bg_pred, 0.999).item()

        return {
            'BackgroundLeakMean': mean_leak,
            'BackgroundLeakP99': p99_leak,
            'BackgroundLeakMax': max_leak,
        }

    def compute_null_x_leak(
        self,
        y_pred: torch.Tensor,
        x_drr: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute null X rule leakage.

        For all samples where first 3 channels of X are null:
        NullXLeakMean = mean(abs(y_pred))
        NullXLeakMax = max(abs(y_pred))

        Args:
            y_pred: Predicted sinogram [B, 1, CP, W]
            x_drr: DRR input [B, C, H, W] with C >= 3

        Returns:
            Dict with keys: NullXLeakMean, NullXLeakMax
        """
        if x_drr.shape[1] < 3:
            return {'NullXLeakMean': 0.0, 'NullXLeakMax': 0.0}

        # Take first 3 channels
        x3 = x_drr[:, :3, :, :]
        x3_norms = x3.abs().sum(dim=(1, 2, 3))  # [B]

        # Find null patches (first 3 channels sum to ~0)
        null_mask = x3_norms < self.eps_x

        if not null_mask.any():
            return {'NullXLeakMean': 0.0, 'NullXLeakMax': 0.0}

        # Get predictions for null samples
        y_pred_null = y_pred[null_mask]
        y_pred_null_abs = torch.abs(y_pred_null)

        mean_leak = y_pred_null_abs.mean().item()
        max_leak = y_pred_null_abs.max().item()

        return {
            'NullXLeakMean': mean_leak,
            'NullXLeakMax': max_leak,
        }

    def compute_occupancy(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute occupancy metrics (punch card detection).

        gt_open = abs(y_true) > eps_bg
        pred_open = abs(y_pred) > t_open

        Returns: Precision_open, Recall_open, F1_open

        Args:
            y_pred: Predicted sinogram [B, 1, CP, W]
            y_true: Ground truth sinogram [B, 1, CP, W]

        Returns:
            Dict with keys: Precision_open, Recall_open, F1_open
        """
        gt_open = torch.abs(y_true) > self.eps_bg
        pred_open = torch.abs(y_pred) > self.t_open

        # Convert to binary (float for computation)
        gt_open = gt_open.float()
        pred_open = pred_open.float()

        # True positives, false positives, false negatives
        tp = (pred_open * gt_open).sum().item()
        fp = (pred_open * (1 - gt_open)).sum().item()
        fn = ((1 - pred_open) * gt_open).sum().item()

        # Precision
        precision = tp / (tp + fp + 1e-8)

        # Recall
        recall = tp / (tp + fn + 1e-8)

        # F1
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            'Precision_open': precision,
            'Recall_open': recall,
            'F1_open': f1,
        }

    def compute_amplitude_error_open(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute amplitude error on open regions.

        MAE_open = mean(|y_pred - y_true| on gt_open)

        Args:
            y_pred: Predicted sinogram [B, 1, CP, W]
            y_true: Ground truth sinogram [B, 1, CP, W]

        Returns:
            Dict with key: MAE_open
        """
        gt_open = torch.abs(y_true) > self.eps_bg

        if not gt_open.any():
            return {'MAE_open': 0.0}

        mae = (y_pred - y_true)[gt_open].abs().mean().item()
        return {'MAE_open': mae}

    def compute_energy_per_cp(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute energy (fluence) per control point metrics.

        Energy_true_cp = sum(y_true[cp,:])
        Energy_pred_cp = sum(y_pred[cp,:])

        Returns: MAE_EnergyCP, Corr_EnergyCP

        Args:
            y_pred: Predicted sinogram [B, 1, CP, W]
            y_true: Ground truth sinogram [B, 1, CP, W]

        Returns:
            Dict with keys: MAE_EnergyCP, Corr_EnergyCP
        """
        # Sum along W dimension: [B, 1, CP, W] -> [B, CP]
        energy_pred = y_pred.squeeze(1).sum(dim=-1)
        energy_true = y_true.squeeze(1).sum(dim=-1)

        # MAE
        mae = (energy_pred - energy_true).abs().mean().item()

        # Correlation (Pearson)
        energy_pred_flat = energy_pred.reshape(-1)
        energy_true_flat = energy_true.reshape(-1)

        pred_mean = energy_pred_flat.mean()
        true_mean = energy_true_flat.mean()

        pred_centered = energy_pred_flat - pred_mean
        true_centered = energy_true_flat - true_mean

        num = (pred_centered * true_centered).sum()
        den = torch.sqrt((pred_centered ** 2).sum() * (true_centered ** 2).sum()).clamp_min(1e-8)
        corr = (num / den).item()
        corr = np.clip(corr, -1.0, 1.0)  # Clamp to [-1, 1]

        return {
            'MAE_EnergyCP': mae,
            'Corr_EnergyCP': corr,
        }

    def compute_stitching_diagnostics(
        self,
        y_full: torch.Tensor,
        denom_acc: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute stitching diagnostics.

        A. DenomMin, DenomP1, DenomMean from denom_acc
        B. SeamScore = 99.9 percentile of |y_full[:,:,cp] - y_full[:,:,cp-1]|

        Args:
            y_full: Reconstructed sinogram [B, 1, CP, W]
            denom_acc: Denominator accumulator from overlap-add [B, 1, CP, W] (optional)

        Returns:
            Dict with keys: DenomMin, DenomP1, DenomMean, SeamScore
        """
        result = {}

        # Denominator health
        if denom_acc is not None:
            denom_flat = denom_acc.reshape(-1)
            result['DenomMin'] = denom_flat.min().item()
            result['DenomP1'] = self._quantile_fp(denom_flat, 0.01).item()
            result['DenomMean'] = denom_flat.mean().item()
        else:
            result['DenomMin'] = 0.0
            result['DenomP1'] = 0.0
            result['DenomMean'] = 0.0

        # Seam discontinuity
        if y_full.shape[2] > 1:  # CP dimension
            seams = []
            for cp in range(1, y_full.shape[2]):
                diff = torch.abs(y_full[:, :, cp, :] - y_full[:, :, cp - 1, :])
                seams.append(diff)

            if seams:
                seams_concat = torch.cat(seams, dim=0).reshape(-1)
                seam_score = self._quantile_fp(seams_concat, 0.999).item()
            else:
                seam_score = 0.0
        else:
            seam_score = 0.0

        result['SeamScore'] = seam_score

        return result

    def compute_all(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x_drr: torch.Tensor,
        y_full: Optional[torch.Tensor] = None,
        denom_acc: Optional[torch.Tensor] = None,
        include_stitching: bool = True
    ) -> Dict[str, float]:
        """
        Compute all sinogram metrics at once.

        Args:
            y_pred: Predicted sinogram [B, 1, CP, W]
            y_true: Ground truth sinogram [B, 1, CP, W]
            x_drr: DRR input [B, C, H, W]
            y_full: Full reconstructed sinogram (for stitching diagnostics)
            denom_acc: Denominator accumulator (for stitching diagnostics)
            include_stitching: Whether to compute stitching metrics

        Returns:
            Dict with all computed metrics
        """
        result = {}

        # 1. Background leakage
        result.update(self.compute_background_leakage(y_pred, y_true))

        # 2. Null X rule
        result.update(self.compute_null_x_leak(y_pred, x_drr))

        # 3. Occupancy
        result.update(self.compute_occupancy(y_pred, y_true))

        # 4. Amplitude error on open regions
        result.update(self.compute_amplitude_error_open(y_pred, y_true))

        # 5. Energy per control point
        result.update(self.compute_energy_per_cp(y_pred, y_true))

        # 6. Stitching diagnostics
        if include_stitching:
            y_full_use = y_full if y_full is not None else y_pred
            result.update(self.compute_stitching_diagnostics(y_full_use, denom_acc))

        return result


def compute_sino_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    x_drr: torch.Tensor,
    denom_acc: Optional[torch.Tensor] = None,
    y_full: Optional[torch.Tensor] = None,
    eps_bg: float = 1e-6,
    t_open: float = 1e-8,
    eps_x: float = 1e-8,
    include_stitching: bool = True
) -> Dict[str, float]:
    """
    High-level function to compute all sinogram metrics.

    This function returns only the essential metrics:
    - BackgroundLeakMean
    - BackgroundLeakP99
    - NullXLeakMax
    - F1_open
    - MAE_open
    - MAE_EnergyCP
    - DenomMin
    - SeamScore

    Args:
        y_pred: Predicted sinogram [B, 1, CP, W]
        y_true: Ground truth sinogram [B, 1, CP, W]
        x_drr: DRR input [B, C, H, W]
        denom_acc: Denominator accumulator from overlap-add (optional)
        y_full: Full reconstructed sinogram (optional, defaults to y_pred)
        eps_bg: Background threshold
        t_open: Opening threshold
        eps_x: Null X threshold
        include_stitching: Whether to compute stitching metrics

    Returns:
        Dict with computed metrics (subset of most important ones)
    """
    metrics_obj = SinogramMetrics(
        eps_bg=eps_bg,
        t_open=t_open,
        eps_x=eps_x,
        device=y_pred.device
    )

    # Compute all metrics
    all_metrics = metrics_obj.compute_all(
        y_pred=y_pred,
        y_true=y_true,
        x_drr=x_drr,
        y_full=y_full,
        denom_acc=denom_acc,
        include_stitching=include_stitching
    )

    # Extract essential metrics
    essential_metrics = {
        'BackgroundLeakMean': all_metrics.get('BackgroundLeakMean', 0.0),
        'BackgroundLeakP99': all_metrics.get('BackgroundLeakP99', 0.0),
        'NullXLeakMax': all_metrics.get('NullXLeakMax', 0.0),
        'F1_open': all_metrics.get('F1_open', 0.0),
        'MAE_open': all_metrics.get('MAE_open', 0.0),
        'MAE_EnergyCP': all_metrics.get('MAE_EnergyCP', 0.0),
    }

    if include_stitching:
        essential_metrics.update({
            'DenomMin': all_metrics.get('DenomMin', 0.0),
            'SeamScore': all_metrics.get('SeamScore', 0.0),
        })

    return essential_metrics

