# ============================================================
# CNN2D Encoder + MLP Binary Classifier (Pass/Fail)
# - Dataset: list of N samples, each is a 3D tensor (B, W, 3)
#   where B and W are variable and MUST NOT be mixed across samples.
# - Training batching: we batch samples by padding to (Bmax, Wmax)
# - Model: Conv2d over spatial (B, W) with 3 input channels
# - Pooling: masked global avg/max pooling -> fixed vector
# - Loss: Focal Loss (logits-based)
# - Validation: logits -> threshold -> pass/fail, then Precision/Recall/F1
# ============================================================

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Metrics
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
# Collate: pad variable (B, W) + mask
# ----------------------------
def collate_bwl_batch(
    batch: List[Tuple[torch.Tensor, int]],
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    batch: list of (X_i, y_i)
      - X_i: FloatTensor (B_i, W_i, 3)  [score, x, y]
      - y_i: int 0/1
    returns:
      X: (Nbatch, Bmax, Wmax, 3)
      mask: (Nbatch, Bmax, Wmax) bool (True=valid)
      y: (Nbatch,)
    """
    xs, ys = zip(*batch)
    nb = len(xs)
    Bmax = max(x.shape[0] for x in xs)
    Wmax = max(x.shape[1] for x in xs)

    X = xs[0].new_full((nb, Bmax, Wmax, 3), fill_value=pad_value)
    mask = xs[0].new_zeros((nb, Bmax, Wmax), dtype=torch.bool)
    y = torch.tensor(ys, dtype=torch.long)

    for i, x in enumerate(xs):
        b, w, c = x.shape
        assert c == 3, f"Expected last dim=3 but got {c}"
        X[i, :b, :w] = x
        mask[i, :b, :w] = True

    return X, mask, y


# ----------------------------
# Focal Loss (binary, logits)
# ----------------------------
class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal loss for binary classification with logits.
    alpha: Optional[float] in [0,1] (pos weight alpha, neg weight 1-alpha) or None
    gamma: focusing parameter
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
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)  # prob of correct class
        focal = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = torch.where(targets > 0.5,
                                  torch.full_like(targets, self.alpha),
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
# Masked pooling for 2D
# ----------------------------
def masked_avg_pool_2d(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    feat: (N, C, B, W)
    mask: (N, B, W) bool
    returns: (N, C)
    """
    mask_f = mask.unsqueeze(1).float()  # (N,1,B,W)
    feat = feat * mask_f
    denom = mask_f.sum(dim=(2, 3)).clamp_min(1.0)  # (N,1)
    return feat.sum(dim=(2, 3)) / denom

def masked_max_pool_2d(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    feat: (N, C, B, W)
    mask: (N, B, W) bool
    returns: (N, C)
    """
    neg_inf = torch.finfo(feat.dtype).min
    mask_exp = mask.unsqueeze(1)  # (N,1,B,W)
    feat = feat.masked_fill(~mask_exp, neg_inf)
    return feat.amax(dim=(2, 3))


# ----------------------------
# Model: CNN2D Encoder + MLP head
# ----------------------------
class ConvBlock2D(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, kernel_size=1, bias=False)

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


class CNN2DPassFail(nn.Module):
    """
    Input:
      X: (N, B, W, 3)  where 3=[score, x, y]
      mask: (N, B, W) bool
    Output:
      logits: (N,)
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

        # 3 input channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, kernel_size=1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ConvBlock2D(base, base, k=k, dropout=dropout) for _ in range(depth)])

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
        # X: (N,B,W,3)
        score = X[..., 0]    # (N,B,W)
        xy = X[..., 1:3]     # (N,B,W,2)

        if self.use_score_log1p:
            score_stable = torch.log1p(torch.clamp(score, min=0.0))
        else:
            score_stable = score

        X2 = torch.cat([score_stable.unsqueeze(-1), xy], dim=-1)  # (N,B,W,3)

        # Conv2d expects (N, C, H, W) => here H=B, W=W
        x = X2.permute(0, 3, 1, 2).contiguous()  # (N,3,B,W)

        x = self.stem(x)
        x = self.blocks(x)

        avg = masked_avg_pool_2d(x, mask)  # (N,base)
        mx  = masked_max_pool_2d(x, mask)  # (N,base)

        feats = [avg, mx]

        if self.add_score_stats:
            # stats over valid cells only
            mask_f = mask.float()
            denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0)  # (N,)

            s_mean = (score_stable * mask_f).sum(dim=(1, 2)) / denom  # (N,)

            neg_inf = torch.finfo(score_stable.dtype).min
            s_max = score_stable.masked_fill(~mask, neg_inf).amax(dim=(1, 2))  # (N,)

            s_feat = self.score_head(torch.stack([s_mean, s_max], dim=1))  # (N,base//2)
            feats.append(s_feat)

        h = torch.cat(feats, dim=1)
        logits = self.head(h).squeeze(-1)  # (N,)
        return logits


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    focal_alpha: Optional[float] = 0.25
    focal_gamma: float = 2.0

    # logits threshold:
    # logits >= 0  <=> sigmoid(logits) >= 0.5
    logit_threshold: float = 0.0


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
        total_loss, n_batches = 0.0, 0

        for X, mask, y in train_loader:
            X = X.to(cfg.device)
            mask = mask.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(X, mask)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)
        history["train_loss"].append(train_loss)

        # ---- Validation (logits -> threshold -> pass/fail) ----
        model.eval()
        val_total_loss, val_batches = 0.0, 0
        all_true, all_pred = [], []

        with torch.no_grad():
            for X, mask, y in val_loader:
                X = X.to(cfg.device)
                mask = mask.to(cfg.device)
                y = y.to(cfg.device)

                logits = model(X, mask)
                loss = criterion(logits, y)

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
# Example dataset wrapper for your "list of tensors"
# ============================================================
class ListTensorDataset(Dataset):
    """
    tensors: List[Tensor] where each Tensor is (B, W, 3)
    labels:  List[int] (0/1)
    """
    def __init__(self, tensors: List[torch.Tensor], labels: List[int]):
        assert len(tensors) == len(labels)
        self.tensors = tensors
        self.labels = labels

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = self.tensors[idx].float()
        y = int(self.labels[idx])
        assert x.dim() == 3 and x.size(-1) == 3, f"Expected (B,W,3), got {tuple(x.shape)}"
        return x, y


# ============================================================
# Minimal runnable demo (remove in your project)
# ============================================================
if __name__ == "__main__":
    # Dummy data: N samples, each is (B,W,3) with variable B/W
    N = 300
    tensors = []
    labels = []
    for _ in range(N):
        B = random.randint(1, 25)
        W = random.randint(1, 40)
        score = torch.rand(B, W) * 10.0          # anomaly score-like
        xy = torch.rand(B, W, 2) * 2.0 - 1.0     # normalized coords example
        x = torch.cat([score.unsqueeze(-1), xy], dim=-1)  # (B,W,3)

        y = 1 if random.random() < 0.15 else 0
        if y == 1:
            # inject a high-score anomaly cell
            bi = random.randrange(B)
            wi = random.randrange(W)
            x[bi, wi, 0] += 20.0
        tensors.append(x)
        labels.append(y)

    # Split
    split = int(0.8 * N)
    train_ds = ListTensorDataset(tensors[:split], labels[:split])
    val_ds = ListTensorDataset(tensors[split:], labels[split:])

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_bwl_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_bwl_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNN2DPassFail(
        base=64,
        depth=4,
        k=3,
        dropout=0.1,
        use_score_log1p=True,
        add_score_stats=True,
    )

    cfg = TrainConfig(
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        focal_alpha=0.25,
        focal_gamma=2.0,
        logit_threshold=0.0,
    )

    history = train_model(model, train_loader, val_loader, cfg)