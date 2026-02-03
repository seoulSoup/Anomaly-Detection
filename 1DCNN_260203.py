# ============================================================
# 1D-CNN (set-like) Encoder + MLP Binary Classifier (Pass/Fail)
# - Input: variable-length set W of 3D vectors [score, x, y]
#   Each sample: (W_i, 3)
# - Batch: padded to max W in batch + mask
# - Model: Conv1d over W dimension + masked global pooling
# - Loss: Focal Loss (logits-based)
# - Train: monitors Precision / Recall / F1 on validation
#   using logits -> criterion(threshold) -> pass/fail
# ============================================================

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Utilities: metrics
# ----------------------------
@torch.no_grad()
def precision_recall_f1_from_preds(
    y_true: torch.Tensor,  # (N,) int {0,1}
    y_pred: torch.Tensor,  # (N,) int {0,1}
    eps: float = 1e-12
) -> Tuple[float, float, float]:
    y_true = y_true.long()
    y_pred = y_pred.long()

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(precision), float(recall), float(f1)


# ----------------------------
# Collate: pad variable W + mask
# ----------------------------
def collate_set_batch(
    batch: List[Tuple[torch.Tensor, int]],
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    batch: list of (X_i, y_i)
      - X_i: Tensor (W_i, 3)
      - y_i: int (0/1)
    returns:
      X: (B, Wmax, 3)
      mask: (B, Wmax) with 1 for valid positions else 0
      y: (B,)
    """
    xs, ys = zip(*batch)
    B = len(xs)
    Wmax = max(x.shape[0] for x in xs)

    X = xs[0].new_full((B, Wmax, 3), fill_value=pad_value)
    mask = xs[0].new_zeros((B, Wmax), dtype=torch.bool)
    y = torch.tensor(ys, dtype=torch.long)

    for i, x in enumerate(xs):
        w = x.shape[0]
        X[i, :w] = x
        mask[i, :w] = True

    return X, mask, y


# ----------------------------
# Focal Loss (binary, logits)
# ----------------------------
class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal loss for binary classification, operating on logits.
    - alpha: class balancing (float in [0,1]) or None
             If float, applied to positive class weight alpha, negative weight (1-alpha)
    - gamma: focusing parameter
    - reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha is not None:
            assert 0.0 <= alpha <= 1.0
        assert gamma >= 0.0
        assert reduction in ("mean", "sum", "none")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,) or (B,1)
        targets: (B,) int/float {0,1}
        """
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        # BCE with logits per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # pt = exp(-bce) is probability of correct class
        pt = torch.exp(-bce)

        # focal modulation
        focal = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(targets > 0.5, torch.full_like(targets, self.alpha),
                                  torch.full_like(targets, 1.0 - self.alpha))
            loss = alpha_t * focal * bce
        else:
            loss = focal * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ----------------------------
# Masked pooling helpers
# ----------------------------
def masked_avg_pool_1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, W)
    mask: (B, W) bool
    returns: (B, C)
    """
    mask_f = mask.unsqueeze(1).float()  # (B,1,W)
    x_masked = x * mask_f
    denom = mask_f.sum(dim=2).clamp_min(1.0)  # (B,1)
    return x_masked.sum(dim=2) / denom

def masked_max_pool_1d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, W)
    mask: (B, W) bool
    returns: (B, C)
    """
    # set padded positions to very negative so max ignores them
    neg_inf = torch.finfo(x.dtype).min
    mask_expand = mask.unsqueeze(1)  # (B,1,W)
    x_masked = x.masked_fill(~mask_expand, neg_inf)
    return x_masked.max(dim=2).values


# ----------------------------
# Model: 1D CNN Encoder + MLP Head
# ----------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel_size=k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel_size=k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = y + self.skip(x)
        y = self.act(y)
        return y


class CNNSetBinaryClassifier(nn.Module):
    """
    Input:
      X: (B, W, 3) where 3=[score, x, y]
      mask: (B, W) bool
    Output:
      logits: (B,)
    Notes:
      - W is a set, but we still apply Conv1d on the padded "W axis".
        Since it's unordered, Conv is a heuristic feature extractor.
        We reduce order-dependence by:
          (a) using only global pooling at the end,
          (b) optionally shuffling W order per sample during training outside the model (recommended).
    """
    def __init__(
        self,
        base: int = 64,
        depth: int = 4,
        k: int = 3,
        dropout: float = 0.1,
        use_score_log1p: bool = True,
        add_score_stats: bool = True,
    ):
        super().__init__()
        self.use_score_log1p = use_score_log1p
        self.add_score_stats = add_score_stats

        self.stem = nn.Sequential(
            nn.Conv1d(3, base, kernel_size=1, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvBlock1D(base, base, k=k, dropout=dropout) for _ in range(depth)])

        score_feat_dim = base // 2 if add_score_stats else 0
        if add_score_stats:
            self.score_head = nn.Sequential(
                nn.Linear(2, score_feat_dim),
                nn.GELU(),
            )

        self.head = nn.Sequential(
            nn.Linear(base * 2 + score_feat_dim, base),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base, 1),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X: (B,W,3), mask: (B,W)
        score = X[..., 0]          # (B,W)
        xy = X[..., 1:3]           # (B,W,2)

        if self.use_score_log1p:
            # If score is non-negative anomaly score, log1p stabilizes large values
            score_stable = torch.log1p(torch.clamp(score, min=0.0))
        else:
            score_stable = score

        X2 = torch.cat([score_stable.unsqueeze(-1), xy], dim=-1)  # (B,W,3)
        x = X2.permute(0, 2, 1).contiguous()  # (B,3,W)

        x = self.stem(x)
        x = self.blocks(x)

        avg = masked_avg_pool_1d(x, mask)  # (B,base)
        mx = masked_max_pool_1d(x, mask)   # (B,base)

        feats = [avg, mx]

        if self.add_score_stats:
            # stats computed only over valid positions
            mask_f = mask.float()
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            s_mean = (score_stable * mask_f).sum(dim=1) / denom  # (B,)
            # max over valid positions
            neg_inf = torch.finfo(score_stable.dtype).min
            s_max = score_stable.masked_fill(~mask, neg_inf).max(dim=1).values  # (B,)
            s_feat = self.score_head(torch.stack([s_mean, s_max], dim=1))       # (B,base//2)
            feats.append(s_feat)

        h = torch.cat(feats, dim=1)
        logits = self.head(h).squeeze(-1)  # (B,)
        return logits


# ----------------------------
# Optional: per-sample set shuffling (recommended)
# ----------------------------
@torch.no_grad()
def shuffle_set_in_batch(X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shuffles the valid W positions per sample to reduce order bias.
    X: (B,W,3), mask: (B,W)
    """
    B, W, C = X.shape
    X_out = X.clone()
    mask_out = mask.clone()
    for i in range(B):
        valid_idx = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= 1:
            continue
        perm = valid_idx[torch.randperm(valid_idx.numel(), device=valid_idx.device)]
        X_out[i, valid_idx] = X[i, perm]
    return X_out, mask_out


# ----------------------------
# Training config + train loop
# ----------------------------
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # focal loss
    focal_alpha: Optional[float] = 0.25
    focal_gamma: float = 2.0

    # decision threshold on logits:
    # logits >= 0  <=> sigmoid(logits) >= 0.5
    logit_threshold: float = 0.0

    # set shuffle augmentation
    shuffle_sets: bool = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig
) -> Dict[str, List[float]]:
    model = model.to(cfg.device)

    criterion = BinaryFocalLossWithLogits(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X, mask, y in train_loader:
            X = X.to(cfg.device)
            mask = mask.to(cfg.device)
            y = y.to(cfg.device)

            if cfg.shuffle_sets:
                X, mask = shuffle_set_in_batch(X, mask)

            logits = model(X, mask)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        # ---- Validation (logits -> threshold -> pass/fail) ----
        model.eval()
        val_total_loss = 0.0
        val_batches = 0

        all_true = []
        all_pred = []

        with torch.no_grad():
            for X, mask, y in val_loader:
                X = X.to(cfg.device)
                mask = mask.to(cfg.device)
                y = y.to(cfg.device)

                logits = model(X, mask)
                loss = criterion(logits, y)

                # Convert logits to pass/fail using threshold on logits directly
                pred = (logits >= cfg.logit_threshold).long()

                val_total_loss += loss.item()
                val_batches += 1
                all_true.append(y.detach().cpu())
                all_pred.append(pred.detach().cpu())

        val_loss = val_total_loss / max(1, val_batches)
        y_true = torch.cat(all_true, dim=0)
        y_pred = torch.cat(all_pred, dim=0)
        p, r, f1 = precision_recall_f1_from_preds(y_true, y_pred)

        history["val_loss"].append(val_loss)
        history["val_precision"].append(p)
        history["val_recall"].append(r)
        history["val_f1"].append(f1)

        print(
            f"[Epoch {epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"P={p:.4f} R={r:.4f} F1={f1:.4f} | thr(logit)={cfg.logit_threshold:.2f}"
        )

    return history


# ============================================================
# Example usage (replace with your Dataset)
# ============================================================
class ExampleSetDataset(Dataset):
    """
    Dummy dataset example.
    Replace __getitem__ to return:
      X_i: FloatTensor (W_i, 3)
      y_i: int 0/1
    """
    def __init__(self, n: int = 1000, w_min: int = 1, w_max: int = 40, pos_rate: float = 0.1):
        super().__init__()
        self.n = n
        self.w_min = w_min
        self.w_max = w_max
        self.pos_rate = pos_rate

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        W = random.randint(self.w_min, self.w_max)
        # score: non-normalized anomaly-like positive values
        score = torch.rand(W) * 10.0
        xy = torch.rand(W, 2) * 2.0 - 1.0
        X = torch.cat([score.unsqueeze(-1), xy], dim=1).float()

        y = 1 if random.random() < self.pos_rate else 0
        # make positives have higher max score (toy signal)
        if y == 1:
            X[random.randrange(W), 0] += 20.0
        return X, y


if __name__ == "__main__":
    # Build loaders
    train_ds = ExampleSetDataset(n=2000, pos_rate=0.12)
    val_ds = ExampleSetDataset(n=500, pos_rate=0.12)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_set_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_set_batch,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = CNNSetBinaryClassifier(
        base=64,
        depth=4,
        k=3,
        dropout=0.1,
        use_score_log1p=True,
        add_score_stats=True,
    )

    # Train
    cfg = TrainConfig(
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        focal_alpha=0.25,
        focal_gamma=2.0,
        logit_threshold=0.0,  # logits>=0 => anomaly(1), else normal(0)
        shuffle_sets=True,
    )

    history = train_model(model, train_loader, val_loader, cfg)