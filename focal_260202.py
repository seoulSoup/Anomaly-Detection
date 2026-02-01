import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary classification using logits.
    - logits: shape [B] or [B,1]
    - targets: float tensor {0,1}, shape [B] or [B,1]
    Params:
      alpha: balance factor for positive class (0~1). If None, no alpha-balancing.
      gamma: focusing parameter (>=0). Typical: 2.0
      reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha: float | None = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)

        targets = targets.to(dtype=logits.dtype)

        # BCE with logits per-sample (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [B]

        # pt = P(correct class)
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)  # [B]

        # focal factor
        focal_factor = (1 - pt).clamp(min=1e-8).pow(self.gamma)

        # alpha balancing
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
                                  torch.tensor(1 - self.alpha, device=logits.device, dtype=logits.dtype))
            loss = alpha_t * focal_factor * bce
        else:
            loss = focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss