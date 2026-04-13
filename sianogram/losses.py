import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GranularSinoLoss(nn.Module):
    """
    Loss orientée sinogrammes "granulaires".

    Composantes:
    1) L1 pondérée par l'intensité cible (renforce les fortes valeurs).
    2) Pénalité quadratique des faux positifs sur fond.
    3) Alignement des gradients sur l'axe des lames (W).
    """

    def __init__(
        self,
        lambda_fp: float = 5.0,
        lambda_grad: float = 1.0,
        lambda_high: float = 2.0,
        gamma_high: float = 2.0,
        tau_low: float = 1e-4,
        apply_sigmoid: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_fp = float(lambda_fp)
        self.lambda_grad = float(lambda_grad)
        self.lambda_high = float(lambda_high)
        self.gamma_high = float(gamma_high)
        self.tau_low = float(tau_low)
        self.apply_sigmoid = bool(apply_sigmoid)
        self.eps = float(eps)

    @staticmethod
    def _gradient_leaf(x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) < 2:
            return torch.zeros_like(x)
        return x[..., 1:] - x[..., :-1]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        del epoch  # compat interface losses existantes

        if self.apply_sigmoid:
            y = torch.sigmoid(y_pred)
        else:
            y = y_pred

        t = y_true.to(y.dtype)

        # 1) Erreur principale pondérée sur les hautes intensités
        weight_high = 1.0 + self.lambda_high * (t.clamp_min(0.0) + self.eps).pow(self.gamma_high)
        base_loss = weight_high * torch.abs(y - t)

        # 2) Faux positifs sur fond (quadratique)
        mask_bg = (t <= self.tau_low).to(y.dtype)
        fp_loss = mask_bg * (y ** 2)

        # 3) Respect des transitions inter-lames
        grad_pred = self._gradient_leaf(y)
        grad_true = self._gradient_leaf(t)
        grad_loss = torch.abs(grad_pred - grad_true)

        total = (
            base_loss.mean()
            + self.lambda_fp * fp_loss.mean()
            + self.lambda_grad * grad_loss.mean()
        )
        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


class BalancedLogSpectralLoss(nn.Module):
    """
    Loss pour sinogrammes [0,1] qui évite l'effet "pics only".
    Aucun terme L1: uniquement MSE (L2) en espace + MSE sur gradients + MSE sur spectre (amplitude).
    Compatible AMP BF16 car FFT forcée en float32.
    """
    def __init__(
        self,
        lam_pix: float = 1.0,
        lam_grad: float = 0.15,
        lam_spec: float = 0.25,
        k_log: float = 20.0,
        w_min: float = 0.20,
        p_w: float = 0.35,
        apply_sigmoid: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lam_pix = lam_pix
        self.lam_grad = lam_grad
        self.lam_spec = lam_spec
        self.k_log = float(k_log)
        self.w_min = float(w_min)
        self.p_w = float(p_w)
        self.apply_sigmoid = apply_sigmoid
        self.eps = float(eps)
        self._log_norm = math.log1p(self.k_log)

    def _log_map(self, x: torch.Tensor) -> torch.Tensor:
        # map [0,1] -> [0,1] mais “étale” les faibles valeurs
        return torch.log1p(self.k_log * x) / self._log_norm

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int = None) -> torch.Tensor:
        # y_pred: [B,1,CP,W] logits ou valeurs
        # y_true: [B,1,CP,W] dans [0,1]
        t = y_true.clamp(0.0, 1.0)

        if self.apply_sigmoid:
            y = torch.sigmoid(y_pred)
        else:
            y = y_pred
        y = y.clamp(0.0, 1.0)

        yl = self._log_map(y)
        tl = self._log_map(t)

        # Pondération douce, avec plancher pour ne pas ignorer le low/mid
        w = self.w_min + (1.0 - self.w_min) * (t + self.eps).pow(self.p_w)

        # Pixel L2 en log-domain
        loss_pix = ((yl - tl).pow(2) * w).mean()

        # Gradients L2 (encourage les bords sans “binariser”)
        dy_cp = yl[:, :, 1:, :] - yl[:, :, :-1, :]
        dt_cp = tl[:, :, 1:, :] - tl[:, :, :-1, :]
        dy_w  = yl[:, :, :, 1:] - yl[:, :, :, :-1]
        dt_w  = tl[:, :, :, 1:] - tl[:, :, :, :-1]
        loss_grad = (dy_cp - dt_cp).pow(2).mean() + (dy_w - dt_w).pow(2).mean()

        # Spectral amplitude en float32 (BF16 FFT pas supporté chez toi)
        device_type = "cuda" if y.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            Yp = torch.fft.rfft2(yl.squeeze(1).float(), dim=(-2, -1))
            Yt = torch.fft.rfft2(tl.squeeze(1).float(), dim=(-2, -1))
            Ap = torch.log1p(torch.abs(Yp))
            At = torch.log1p(torch.abs(Yt))
            loss_spec = F.mse_loss(Ap, At)

        return self.lam_pix * loss_pix + self.lam_grad * loss_grad + self.lam_spec * loss_spec

class SparseFocalSpectralLoss(nn.Module):
    """
    Loss pour sinogrammes sparsifiés, orientée "pixel art", sans L1.

    Composantes:
    1) Weighted MSE amplitude (zones actives >> fond)
    2) Focal BCE sur la "présence" (mask) dérivée de y_pred via un logit k*(y - tau)
    3) MSE sur gradients (CP et W) pour pousser les détails
    4) MSE sur spectre (log amplitude FFT) pour pousser les hautes fréquences

    Entrées attendues: y_pred, y_true en [B, 1, CP, W]
    """
    def __init__(
        self,
        thr: float = 0.02,              # seuil pour définir "actif" sur y_true
        apply_sigmoid: bool = True,     # utile si ton modèle ne borne pas en [0,1]

        # Weighted MSE
        w_active: float = 6.0,          # poids de base des pixels actifs
        w_bg: float = 0.2,              # poids du fond (évite d'ignorer totalement le fond)
        intensity_gamma: float = 2.0,   # surpoids des pics via (y_true^gamma)

        # Faux positifs sur fond (quadratique)
        w_fp: float = 4.0,
        fp_margin: float = 0.0,

        # Masque (Focal BCE)
        w_mask: float = 1.0,
        mask_tau: float = 0.02,         # seuil "actif" côté prédiction
        mask_k: float = 40.0,           # raideur du logit: logit = k*(y - tau)
        focal_gamma: float = 2.0,

        # Hautes fréquences
        w_grad_cp: float = 0.2,
        w_grad_w: float = 0.2,
        w_fft: float = 0.05,

        # Rampe (éviter collapse au début)
        ramp_epochs: int = 10,
        ramp_power: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.thr = float(thr)
        self.apply_sigmoid = bool(apply_sigmoid)

        self.w_active = float(w_active)
        self.w_bg = float(w_bg)
        self.intensity_gamma = float(intensity_gamma)

        self.w_fp = float(w_fp)
        self.fp_margin = float(fp_margin)

        self.w_mask = float(w_mask)
        self.mask_tau = float(mask_tau)
        self.mask_k = float(mask_k)
        self.focal_gamma = float(focal_gamma)

        self.w_grad_cp = float(w_grad_cp)
        self.w_grad_w = float(w_grad_w)
        self.w_fft = float(w_fft)

        self.ramp_epochs = int(max(0, ramp_epochs))
        self.ramp_power = float(ramp_power)
        self.eps = float(eps)

    def _ramp(self, epoch: int | None) -> float:
        if self.ramp_epochs <= 0 or epoch is None:
            return 1.0
        t = min(max(int(epoch), 0), self.ramp_epochs) / float(self.ramp_epochs)
        t = t ** self.ramp_power
        return float(t)

    @staticmethod
    def _diff(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.size(dim) < 2:
            return torch.zeros_like(x)
        sl1 = [slice(None)] * x.ndim
        sl2 = [slice(None)] * x.ndim
        sl1[dim] = slice(1, None)
        sl2[dim] = slice(0, -1)
        return x[tuple(sl1)] - x[tuple(sl2)]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int | None = None):
        if self.apply_sigmoid:
            y = torch.sigmoid(y_pred)
        else:
            y = y_pred

        y_true = y_true.to(y.dtype)

        # 1) Weighted MSE amplitude
        M = (y_true > self.thr).to(y.dtype)
        invM = 1.0 - M

        with torch.no_grad():
            amp_w = (y_true.clamp_min(0.0) ** self.intensity_gamma)
            w = self.w_bg * invM + self.w_active * M * (1.0 + amp_w)

        err2 = (y - y_true) ** 2
        loss_amp = (w * err2).sum() / w.sum().clamp_min(1.0)

        # 2) Faux positifs fond (quadratique, pas L1)
        fp = torch.relu(y - self.fp_margin) * invM
        loss_fp = (fp ** 2).sum() / invM.sum().clamp_min(1.0)

        # 3) Focal BCE sur masque "actif" (utilise un logit dérivé de y)
        # logit = k*(y - tau) => p = sigmoid(logit)
        logit = self.mask_k * (y - self.mask_tau)
        tgt = M
        bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        p = torch.sigmoid(logit)
        pt = tgt * p + (1.0 - tgt) * (1.0 - p)
        focal = ((1.0 - pt).clamp_min(0.0) ** self.focal_gamma) * bce
        loss_mask = focal.mean()

        # 4) Gradients (MSE)
        dy_cp = self._diff(y, dim=2)
        dt_cp = self._diff(y_true, dim=2)
        dy_w = self._diff(y, dim=3)
        dt_w = self._diff(y_true, dim=3)

        # masque gradient: actif si l'un des deux pixels voisins est actif
        M_cp = (self._diff(M, dim=2).abs() > 0).to(y.dtype)
        M_w = (self._diff(M, dim=3).abs() > 0).to(y.dtype)

        loss_grad_cp = (((dy_cp - dt_cp) ** 2) * (1.0 + 2.0 * M_cp)).mean()
        loss_grad_w = (((dy_w - dt_w) ** 2) * (1.0 + 2.0 * M_w)).mean()

        # 5) Spectral (log amplitude FFT) MSE
        # log1p pour stabiliser la dynamique des pics
        Yp = torch.fft.rfft2(y.squeeze(1), dim=(-2, -1))
        Yt = torch.fft.rfft2(y_true.squeeze(1), dim=(-2, -1))
        Ap = torch.log1p(torch.abs(Yp))
        At = torch.log1p(torch.abs(Yt))
        loss_fft = ((Ap - At) ** 2).mean()

        # Rampe: au début on évite de sur-penaliser la sparsité et les HF
        r = self._ramp(epoch)
        w_fp = self.w_fp * r
        w_mask = self.w_mask * r
        w_grad_cp = self.w_grad_cp * r
        w_grad_w = self.w_grad_w * r
        w_fft = self.w_fft * r

        total = (
            loss_amp
            + w_fp * loss_fp
            + w_mask * loss_mask
            + w_grad_cp * loss_grad_cp
            + w_grad_w * loss_grad_w
            + w_fft * loss_fft
        )

        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

class SparseSinoLoss(nn.Module):
    """
    Loss sparse-friendly pour sinogrammes (fond ~0 ultra-majoritaire).

    - loss_pos : précision sur zones actives (y_true > thr)
    - loss_neg : pénalité faux-positifs sur fond (y_pred > neg_margin quand y_true~0)
    - ramp : montée progressive de w_neg pour éviter l'effondrement "tout à 0" au début
    """
    def __init__(
        self,
        thr: float = 0.02,
        w_pos: float = 1.0,
        w_neg: float = 8.0,

        neg_power: float = 2.0,
        neg_margin: float = 0.00,

        huber_beta: float = 0.05,
        use_sigmoid: bool = False,

        # ---- RAMPE ----
        ramp_epochs: int = 10,      # <-- RAMPE PLUS COURTE (avant 20-50)
        ramp_power: float = 0.5,    # <-- <1 = montée plus rapide au début (sqrt)
    ):
        super().__init__()
        self.thr = float(thr)
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)

        self.neg_power = float(neg_power)
        self.neg_margin = float(neg_margin)

        self.huber_beta = float(huber_beta)
        self.use_sigmoid = bool(use_sigmoid)

        self.ramp_epochs = int(ramp_epochs)
        self.ramp_power = float(ramp_power)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epoch: int | None = None):
        # y_pred, y_true: [B,1,Hcp,W]
        if self.use_sigmoid:
            y = torch.sigmoid(y_pred)
        else:
            y = y_pred

        M = (y_true > self.thr).to(y.dtype)
        invM = 1.0 - M

        # --- Zones actives ---
        pos_sum = F.smooth_l1_loss(y * M, y_true * M, beta=self.huber_beta, reduction="sum")
        pos_den = M.sum().clamp_min(1.0)
        loss_pos = pos_sum / pos_den

        # --- Fond (faux-positifs) ---
        fp = torch.relu(y - self.neg_margin) * invM
        neg_sum = (fp.abs() ** self.neg_power).sum()
        neg_den = invM.sum().clamp_min(1.0)
        loss_neg = neg_sum / neg_den

        # --- Rampe courte ---
        if self.ramp_epochs > 0 and epoch is not None:
            t = min(max(int(epoch), 0), self.ramp_epochs) / float(self.ramp_epochs)
            # montée plus rapide au début si ramp_power < 1
            t = t ** self.ramp_power
            w_neg = self.w_neg * t
        else:
            w_neg = self.w_neg

        return self.w_pos * loss_pos + w_neg * loss_neg


class GigaUltimateLoss(nn.Module):
    """
    Loss sinogramme robuste et interprétable.

    Objectif clé: éviter que les très nombreux pixels à 0 (fond) dominent l'optimisation.
    Pour cela, la L1 est pondérée: les zones actives (target > thr) comptent davantage.

    L = w_l1 * L1_weighted
        + ramp * ( w_grad_w * GradW + w_grad_cp * GradCP
                   + w_fluence_cp * FluenceCP + w_leaf_fluence * LeafFluence
                   + w_ffl_x * FFL_X + w_ffl2d * FFL_2D )

    Notes:
    pred/target attendus en [B, C, CP, W] (ou équivalent, du moment que les deux
    derniers axes sont CP et W).
    """
    def __init__(
        self,
        w_l1: float = 1.0,

        use_weighted_l1: bool = True,
        nonzero_thr: float = 0.02,
        nonzero_weight: float = 6.0,
        intensity_weight: float = 0.0,
        intensity_power: float = 1.0,

        w_grad_w: float = 0.0,
        w_grad_cp: float = 0.0,
        w_fluence_cp: float = 0.0,
        w_leaf_fluence: float = 0.0,

        w_ffl_x: float = 0.0,
        w_ffl2d: float = 0.0,
        ffl_p: float = 2.0,

        ramp_epochs: int = 10,
        eps: float = 1e-6
    ):
        super().__init__()
        self.w_l1 = float(w_l1)

        self.use_weighted_l1 = bool(use_weighted_l1)
        self.nonzero_thr = float(nonzero_thr)
        self.nonzero_weight = float(nonzero_weight)
        self.intensity_weight = float(intensity_weight)
        self.intensity_power = float(intensity_power)

        self.w_grad_w = float(w_grad_w)
        self.w_grad_cp = float(w_grad_cp)
        self.w_fluence_cp = float(w_fluence_cp)
        self.w_leaf_fluence = float(w_leaf_fluence)

        self.w_ffl_x = float(w_ffl_x)
        self.w_ffl2d = float(w_ffl2d)
        self.ffl_p = float(ffl_p)

        self.ramp_epochs = int(max(1, ramp_epochs))
        self.eps = float(eps)

    @staticmethod
    def _safe(x: torch.Tensor, v: float = 0.0) -> torch.Tensor:
        return torch.nan_to_num(x, nan=v, posinf=v, neginf=v)

    def _ramp(self, epoch: int | None) -> float:
        if epoch is None:
            return 1.0
        return float(min(1.0, max(0.0, epoch / self.ramp_epochs)))

    def _finite_diff(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.size(dim) < 2:
            return torch.zeros_like(x)
        sl_from = [slice(None)] * x.ndim
        sl_to = [slice(None)] * x.ndim
        sl_from[dim] = slice(1, None)
        sl_to[dim] = slice(0, -1)
        return x[tuple(sl_from)] - x[tuple(sl_to)]

    def _weighted_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = torch.abs(pred - target)

        if not self.use_weighted_l1:
            return err.mean()

        with torch.no_grad():
            active = (target > self.nonzero_thr).to(err.dtype)
            w = 1.0 + self.nonzero_weight * active

            if self.intensity_weight > 0.0:
                t = target.clamp(0.0, 1.0)
                w = w + self.intensity_weight * (t ** self.intensity_power)

            w = w.clamp_min(1.0)

        num = (w * err).sum()
        den = w.sum().clamp_min(self.eps)
        return num / den

    def _grad_w_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gp = self._finite_diff(pred, dim=-1)
        gt = self._finite_diff(target, dim=-1)
        return torch.mean(torch.abs(gp - gt))

    def _grad_cp_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gp = self._finite_diff(pred, dim=-2)
        gt = self._finite_diff(target, dim=-2)
        return torch.mean(torch.abs(gp - gt))

    def _fluence_cp_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = pred.sum(dim=-1)
        ft = target.sum(dim=-1)
        return torch.mean(torch.abs(fp - ft))

    def _leaf_fluence_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lp = pred.sum(dim=-2)
        lt = target.sum(dim=-2)
        return torch.mean(torch.abs(lp - lt))

    def _ffl_x(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        F_diff = torch.fft.rfft(diff, dim=-1, norm="ortho")
        Wr = F_diff.shape[-1]

        fx = torch.fft.rfftfreq(n=(Wr - 1) * 2, d=1.0, device=diff.device).view(1, 1, 1, Wr)
        w = (fx / fx.max().clamp_min(self.eps)) ** self.ffl_p

        mag = torch.view_as_real(F_diff)
        mag2 = mag[..., 0] ** 2 + mag[..., 1] ** 2
        return (w * mag2).mean()

    def _ffl_2d(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        F_diff = torch.fft.rfft2(diff, norm="ortho")
        H = diff.shape[-2]
        Wr = F_diff.shape[-1]

        fy = torch.fft.fftfreq(H, d=1.0, device=diff.device).view(1, 1, H, 1)
        fx = torch.fft.rfftfreq(n=(Wr - 1) * 2, d=1.0, device=diff.device).view(1, 1, 1, Wr)
        freq = torch.sqrt(fy ** 2 + fx ** 2)
        w = (freq / freq.max().clamp_min(self.eps)) ** self.ffl_p

        mag = torch.view_as_real(F_diff)
        mag2 = mag[..., 0] ** 2 + mag[..., 1] ** 2
        return (w * mag2).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        pred = self._safe(pred, 0.0)
        target = self._safe(target, 0.0)

        base_l1 = self._safe(self._weighted_l1(pred, target))
        loss = self.w_l1 * base_l1

        if (
            self.w_grad_w == 0.0 and self.w_grad_cp == 0.0 and
            self.w_fluence_cp == 0.0 and self.w_leaf_fluence == 0.0 and
            self.w_ffl_x == 0.0 and self.w_ffl2d == 0.0
        ):
            return loss

        ramp = self._ramp(epoch)

        if self.w_grad_w:
            loss = loss + ramp * self.w_grad_w * self._safe(self._grad_w_loss(pred, target))
        if self.w_grad_cp:
            loss = loss + ramp * self.w_grad_cp * self._safe(self._grad_cp_loss(pred, target))
        if self.w_fluence_cp:
            loss = loss + ramp * self.w_fluence_cp * self._safe(self._fluence_cp_loss(pred, target))
        if self.w_leaf_fluence:
            loss = loss + ramp * self.w_leaf_fluence * self._safe(self._leaf_fluence_loss(pred, target))
        if self.w_ffl_x:
            loss = loss + ramp * self.w_ffl_x * self._safe(self._ffl_x(pred, target))
        if self.w_ffl2d:
            loss = loss + ramp * self.w_ffl2d * self._safe(self._ffl_2d(pred, target))

        return loss

class TomoSinoStrictZeroLoss(nn.Module):
    """
    Loss stricte pour sinogrammes Tomo avec contraintes:
    1. Fond doit être exactement 0 (aucun bruit toléré)
    2. Si X_drr entièrement nul => y_pred strictement nul (gating)
    3. Signal parcimonieux (type "carton perforé"), pics préservés via Huber

    Stratégie:
    - Huber (SmoothL1) sur zones signal uniquement (moins lissage des pics)
    - MSE sur fond uniquement vers 0 (pénalité très forte)
    - MSE "gating" pour X==0 => y==0 (pénalité très forte)
    - Bonus: pondération amplitude optionnelle pour sur-peser les pics

    API:
        loss, metrics = forward(y_pred, y_true, x_input)
        où metrics = dict(loss_sig, loss_bg, loss_gate, frac_sig, ...)
    """
    def __init__(
        self,
        eps_y: float = 1e-6,          # seuil signal/fond
        eps_x: float = 1e-8,          # seuil patch "entièrement nul"
        lambda_bg: float = 50.0,      # poids du fond
        lambda_gate: float = 200.0,   # poids du gating X==0
        beta: float = 0.01,           # Huber beta (petit = proche L1 autour pics)
        use_l1_bg: bool = False,      # False: MSE; True: L1 pour fond (plus sparse)
        use_amp_weight: bool = False, # Bonus: pondération amplitude
        alpha: float = 2.0,           # intensité surpoids (si use_amp_weight)
        p: float = 2.0,               # exposant (si use_amp_weight)
    ):
        super().__init__()
        self.eps_y = float(eps_y)
        self.eps_x = float(eps_x)
        self.lambda_bg = float(lambda_bg)
        self.lambda_gate = float(lambda_gate)
        self.beta = float(beta)
        self.use_l1_bg = bool(use_l1_bg)
        self.use_amp_weight = bool(use_amp_weight)
        self.alpha = float(alpha)
        self.p = float(p)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            y_pred: [B, 1, Lcp, W] ou [B, Lcp, W] - prédiction (peut être ~unbounded)
            y_true: [B, 1, Lcp, W] ou [B, Lcp, W] - ground truth sinogramme
            x_input: [B, C>=3, Hcp, W] - input DRR pour détecter patches nulls (optionnel)

        Returns:
            (loss_total, metrics_dict)
        """
        # Ensure shapes [B, 1, Lcp, W]
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        B, _, Lcp, W = y_true.shape
        device = y_pred.device

        # ========== Masques Signal/Fond ==========
        mask_sig = (torch.abs(y_true) > self.eps_y).float()
        mask_bg = 1.0 - mask_sig

        # ========== Détection patch X==0 (gating) ==========
        if x_input is not None:
            # x_input: [B, C, Hcp, W] - on prend les 3 premiers canaux
            x3 = x_input[:, :min(3, x_input.shape[1]), :, :]  # [B, <=3, Hcp, W]
            # g = 1 si sum(abs(x3)) < eps_x (patch entièrement nul)
            g = (x3.abs().sum(dim=(1, 2, 3), keepdim=False) < self.eps_x).float()  # [B]
            g_map = g.view(B, 1, 1, 1)  # Broadcast-ready
        else:
            g = torch.zeros(B, device=device)
            g_map = torch.zeros(B, 1, 1, 1, device=device)

        # ========== Loss Signal: Huber (SmoothL1) ==========
        n_sig = mask_sig.sum().item()
        if n_sig > 0:
            sig_pred = y_pred * mask_sig
            sig_true = y_true * mask_sig

            if self.use_amp_weight:
                # Pondération amplitude sur signal
                with torch.no_grad():
                    # w = 1 + alpha * (|y_true| / max(|y_true|))^p
                    y_true_abs = torch.abs(sig_true).clamp_min(1e-8)
                    y_max = y_true_abs.max().clamp_min(1e-8)
                    w_amp = 1.0 + self.alpha * (y_true_abs / y_max) ** self.p
                    w_amp = w_amp * mask_sig

                loss_sig = F.smooth_l1_loss(sig_pred, sig_true, beta=self.beta, reduction="none")
                loss_sig = (w_amp * loss_sig).sum() / w_amp.sum().clamp_min(1e-8)
            else:
                loss_sig = F.smooth_l1_loss(sig_pred, sig_true, beta=self.beta, reduction="mean")
        else:
            loss_sig = torch.tensor(0.0, device=device, dtype=y_pred.dtype)

        # ========== Loss Fond: MSE ou L1 ==========
        n_bg = mask_bg.sum().item()
        if n_bg > 0:
            bg_pred = y_pred * mask_bg

            if self.use_l1_bg:
                loss_bg = (torch.abs(bg_pred)).sum() / n_bg
            else:
                loss_bg = (bg_pred ** 2).sum() / n_bg
        else:
            loss_bg = torch.tensor(0.0, device=device, dtype=y_pred.dtype)

        # ========== Loss Gating: X==0 => y==0 ==========
        # Pénaliser fortement y_pred globalement si patch est entièrement nul
        loss_gate_per_batch = (g_map * (y_pred ** 2)).mean(dim=(1, 2, 3), keepdim=False)  # [B]
        loss_gate = loss_gate_per_batch.mean()

        # ========== Loss Totale ==========
        loss_total = loss_sig + self.lambda_bg * loss_bg + self.lambda_gate * loss_gate

        # ========== Métriques pour logging ==========
        metrics = {
            "loss_sig": loss_sig.item() if torch.isfinite(loss_sig) else 0.0,
            "loss_bg": loss_bg.item() if torch.isfinite(loss_bg) else 0.0,
            "loss_gate": loss_gate.item() if torch.isfinite(loss_gate) else 0.0,
            "frac_sig": (n_sig / (Lcp * W)) if (Lcp * W) > 0 else 0.0,
            "n_null_patches": g.sum().item(),
        }

        return loss_total, metrics
