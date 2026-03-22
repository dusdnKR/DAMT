import torch
import torch.nn as nn
from torch.nn import functional as F


def _off_diagonal(x):
    """Return all off-diagonal elements of a square matrix as a 1-D vector."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICReg(nn.Module):
    """Variance-Invariance-Covariance Regularization (VICReg).

    A negative-free self-supervised objective that works well at small batch sizes.
    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning", ICLR 2022.

    Three terms:
      Invariance  : MSE between the two views' embeddings (push paired reps together)
      Variance    : hinge on per-feature std to prevent dimensional collapse
      Covariance  : penalise off-diagonal covariance to reduce redundancy

    Embeddings from all GPUs are gathered before computing variance/covariance
    so the statistics are estimated on the full effective batch.
    """

    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0,
                 cov_coeff: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    @staticmethod
    def _all_gather(tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor from all DDP ranks; returns local tensor if not distributed."""
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return tensor
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def forward(self, z: torch.Tensor, z_prime: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z, z_prime: projected embeddings for two augmented views, shape (N, D).
        Returns scalar loss.
        """
        # Gather across all GPUs for reliable variance / covariance estimation
        z_all       = self._all_gather(z)
        z_prime_all = self._all_gather(z_prime)
        N, D = z_all.shape

        # ── Invariance ───────────────────────────────────────────────────────
        sim_loss = F.mse_loss(z_all, z_prime_all)

        # ── Variance ─────────────────────────────────────────────────────────
        std_z  = torch.sqrt(z_all.var(dim=0)       + 1e-4)
        std_zp = torch.sqrt(z_prime_all.var(dim=0) + 1e-4)
        var_loss = (F.relu(self.gamma - std_z).mean() +
                    F.relu(self.gamma - std_zp).mean()) / 2

        # ── Covariance ────────────────────────────────────────────────────────
        z_c   = z_all       - z_all.mean(dim=0)
        zp_c  = z_prime_all - z_prime_all.mean(dim=0)
        cov_z  = (z_c.T  @ z_c)  / (N - 1)
        cov_zp = (zp_c.T @ zp_c) / (N - 1)
        cov_loss = (_off_diagonal(cov_z).pow(2).sum() +
                    _off_diagonal(cov_zp).pow(2).sum()) / D

        return (self.sim_coeff * sim_loss
                + self.std_coeff * var_loss
                + self.cov_coeff * cov_loss)


class AutoWeightedLoss(nn.Module):
    """Uncertainty-based automatic multi-task loss weighting.

    Learns a log-variance parameter per task so that
        total = sum_i  exp(-log_var_i) * loss_i  +  log_var_i

    Reference: Kendall et al., "Multi-task learning using uncertainty to
    weigh losses for scene geometry and semantics", CVPR 2018.

    Args:
        task_names: Ordered list of task name strings.  The same order must
            be used when passing losses to ``forward()``.
    """

    TASK_NAMES = ["rot", "loc", "contrastive", "atlas", "feat", "texture", "mim", "msn", "asym"]

    def __init__(self, task_names=None):
        super().__init__()
        if task_names is None:
            task_names = self.TASK_NAMES
        self.task_names = list(task_names)
        n = len(self.task_names)
        # Initialise log-variances to 0  →  initial weight = exp(0) = 1
        self.log_vars = nn.Parameter(torch.zeros(n))

    def forward(self, loss_dict: dict):
        """Compute the weighted total loss.

        Args:
            loss_dict: ``{task_name: scalar_loss}`` for every active task.
                Values that are ``0`` (int) are treated as inactive and skipped.

        Returns:
            total_loss: Weighted scalar loss for back-propagation.
            weighted_dict: ``{task_name: weighted_loss_value}`` for logging.
        """
        total = torch.tensor(0.0, device=self.log_vars.device)
        weighted_dict = {}
        for i, name in enumerate(self.task_names):
            raw = loss_dict.get(name, 0)
            if isinstance(raw, (int, float)) and raw == 0:
                weighted_dict[name] = 0.0
                continue
            precision = torch.exp(-self.log_vars[i])
            w_loss = precision * raw + self.log_vars[i]
            total = total + w_loss
            weighted_dict[name] = w_loss.item()
        return total, weighted_dict

    def get_weights(self):
        """Return current effective weights (exp(-log_var)) as a dict."""
        with torch.no_grad():
            weights = torch.exp(-self.log_vars)
        return {name: weights[i].item() for i, name in enumerate(self.task_names)}
