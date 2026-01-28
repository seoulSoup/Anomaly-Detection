# ASL (Asymmetric Loss) + Gradient Accumulation + Sample-level Oversampling (PyTorch Geometric)
# - Each sample = one matrix/graph, one label (0 vs non-0).
# - No "concatenating matrices under each other" inside a sample.
# - Oversampling is done at *sample selection* level (sampler), not inside the matrix.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler

from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, GlobalAttention


# -----------------------------
# 1) Dataset wrapper (if needed)
# -----------------------------
class GraphDataset(Dataset):
    """
    Holds a list of PyG Data objects.
    Each Data must have:
      - data.x: node features (N, F)
      - data.edge_index: (2, E)
      - data.y: scalar label, either {0,1,2,3} or already binary {0,1}
    """
    def __init__(self, data_list: List[Data], assume_multiclass_y: bool = True):
        self.data_list = data_list
        self.assume_multiclass_y = assume_multiclass_y

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        d = self.data_list[idx]
        return d

    @torch.no_grad()
    def binary_labels(self) -> torch.Tensor:
        """
        Return binary labels per sample: 1 if y != 0 else 0.
        """
        ys = []
        for d in self.data_list:
            y = d.y
            # y might be tensor scalar or Python int
            if torch.is_tensor(y):
                yv = int(y.view(-1)[0].item())
            else:
                yv = int(y)
            ys.append(1 if yv != 0 else 0)
        return torch.tensor(ys, dtype=torch.long)


# -------------------------------------------
# 2) Sample-level oversampling (no structure change)
# -------------------------------------------
def make_weighted_sampler_from_binary_labels(y_bin: torch.Tensor) -> WeightedRandomSampler:
    """
    y_bin: (N,) long tensor with 0/1 labels for each sample.
    Oversamples minority class by inverse-frequency weights.
    """
    # counts[0]=neg, counts[1]=pos
    counts = torch.bincount(y_bin, minlength=2).float()
    counts = torch.clamp(counts, min=1.0)

    # weight per sample = 1 / count[label]
    w = 1.0 / counts[y_bin].float()
    sampler = WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
    return sampler


# -----------------------------
# 3) ASL (binary) with logits
# -----------------------------
class AsymmetricLossBinary(nn.Module):
    """
    Asymmetric Loss (ASL) adapted for binary classification.

    Typical setting (start point):
      gamma_pos = 1.0
      gamma_neg = 4.0
      clip = 0.05  (optional; helps suppress easy negatives)
    """
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,) or (B,1) raw logits
        targets: (B,) float tensor with 0/1
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # probabilities
        p = torch.sigmoid(logits)

        # ASL "clipping" for negatives: increase p for negatives slightly (reduces easy-neg gradients)
        # Many ASL implementations clip (1 - p) or p_neg; here we clip p for negatives.
        if self.clip is not None and self.clip > 0:
            p_neg = torch.clamp(p + self.clip, max=1.0)  # only used for negative term below
        else:
            p_neg = p

        # basic log-loss terms
        # positive: -log(p)
        # negative: -log(1 - p)
        pos_loss = -torch.log(torch.clamp(p, min=self.eps))
        neg_loss = -torch.log(torch.clamp(1.0 - p_neg, min=self.eps))

        # focal/asymmetric focusing
        # for pos: (1 - p) ^ gamma_pos
        # for neg: p ^ gamma_neg  (hard negatives have larger p, hence higher weight)
        pos_weight = torch.pow(1.0 - p, self.gamma_pos)
        neg_weight = torch.pow(p, self.gamma_neg)

        loss = targets * pos_weight * pos_loss + (1.0 - targets) * neg_weight * neg_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"


# -----------------------------
# 4) Simple GraphSAGE + Attention Pooling (binary logit)
# -----------------------------
class SageAttnBinary(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, hidden))
        else:
            self.convs.append(SAGEConv(in_dim, hidden))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden, hidden))

        self.dropout = dropout
        self.act = nn.GELU()

        # Attention pooling over nodes -> graph embedding
        self.att = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1)
            )
        )

        # Binary classifier head (1 logit)
        self.head = nn.Linear(hidden, 1)

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)

        if batch is None:
            # single graph fallback
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = self.att(x, batch)              # (num_graphs, hidden)
        logit = self.head(g).view(-1)       # (num_graphs,)
        return logit


# -----------------------------
# 5) Train/eval utilities
# -----------------------------
@torch.no_grad()
def eval_logits_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits: (N,)
      scores: sigmoid(logits) (N,)
      y_bin: (N,) long {0,1}
    """
    model.eval()
    all_logits, all_scores, all_y = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)  # (num_graphs,)
        scores = torch.sigmoid(logits)

        # binary label per graph
        y = batch.y.view(-1)
        y_bin = (y != 0).long()

        all_logits.append(logits.detach().cpu())
        all_scores.append(scores.detach().cpu())
        all_y.append(y_bin.detach().cpu())

    return torch.cat(all_logits), torch.cat(all_scores), torch.cat(all_y)


@torch.no_grad()
def best_f1_threshold_fast(scores: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """
    O(N log N) best F1 threshold from scores (0..1) and labels (0/1).
    Returns: (best_f1, best_thr)
    """
    scores = scores.float().view(-1)
    labels = labels.long().view(-1)

    P = int(labels.sum().item())
    N = labels.numel()
    if P == 0 or N == 0:
        return 0.0, 1.0

    order = torch.argsort(scores, descending=True)
    s = scores[order]
    y = labels[order]

    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    fn = P - tp

    eps = 1e-9
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)

    idx = int(torch.argmax(f1).item())
    return float(f1[idx].item()), float(s[idx].item())


@dataclass
class TrainConfig:
    batch_size: int = 1               # keep 1 if you want; PyG batching doesn't break structure though
    accum_steps: int = 32             # gradient accumulation steps
    max_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0

    # ASL params
    gamma_pos: float = 1.0
    gamma_neg: float = 4.0
    clip: float = 0.05

    # monitoring
    eval_every: int = 1               # evaluate val every N epochs
    early_patience: int = 30
    min_delta: float = 1e-4


def train_asl_accum_oversample(
    model: nn.Module,
    train_dataset: GraphDataset,
    val_dataset: GraphDataset,
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    Uses:
      - sample-level oversampling (WeightedRandomSampler) on train_dataset
      - gradient accumulation to simulate larger batch updates
      - ASL loss on binary labels
      - best weight selection based on val best-F1 (threshold chosen on val)
    """
    model = model.to(device)

    # --- head bias init (ONE-TIME) ---
    with torch.no_grad():
        if hasattr(model, "head") and isinstance(model.head, nn.Linear) and model.head.bias is not None:
            model.head.bias.fill_(1.0)

    # --- samplers/loaders ---
    y_train_bin = train_dataset.binary_labels()
    sampler = make_weighted_sampler_from_binary_labels(y_train_bin)

    def pyg_collate_fn(items: List[Data]) -> Batch:
        return Batch.from_data_list(items)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=pyg_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(knowable_batch_size(cfg.batch_size), 1),
        shuffle=False,
        collate_fn=pyg_collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # --- loss & optimizer ---
    criterion = AsymmetricLossBinary(
        gamma_pos=cfg.gamma_pos,
        gamma_neg=cfg.gamma_neg,
        clip=cfg.clip,
        reduction="mean",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_f1 = -1.0
    best_thr = 0.5
    wait = 0

    global_step = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        step_in_epoch = 0

        for batch in train_loader:
            step_in_epoch += 1
            global_step += 1

            batch = batch.to(device)
            logits = model(batch).view(-1)                 # (num_graphs,)
            y = (batch.y.view(-1) != 0).float()            # binary float 0/1 per graph

            loss = criterion(logits, y)
            loss = loss / cfg.accum_steps
            loss.backward()

            running_loss += float(loss.item())

            if (global_step % cfg.accum_steps) == 0:
                if cfg.clip_grad_norm is not None and cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # flush leftover grads if epoch ends mid-accum
        if (global_step % cfg.accum_steps) != 0:
            if cfg.clip_grad_norm is not None and cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, step_in_epoch)

        # --- validation ---
        if (epoch % cfg.eval_every) == 0:
            val_logits, val_scores, val_y = eval_logits_scores(model, val_loader, device)
            val_f1, thr = best_f1_threshold_fast(val_scores, val_y)

            # also monitor separation quickly
            pos = val_y == 1
            neg = val_y == 0
            pos_mean = float(val_scores[pos].mean().item()) if pos.any() else float("nan")
            neg_mean = float(val_scores[neg].mean().item()) if neg.any() else float("nan")
            pos_max = float(val_scores[pos].max().item()) if pos.any() else float("nan")
            neg_max = float(val_scores[neg].max().item()) if neg.any() else float("nan")

            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"val_bestF1={val_f1:.4f} thr={thr:.4f} "
                f"pos(mean/max)={pos_mean:.4g}/{pos_max:.4g} "
                f"neg(mean/max)={neg_mean:.4g}/{neg_max:.4g}"
            )

            if val_f1 > best_val_f1 + cfg.min_delta:
                best_val_f1 = val_f1
                best_thr = thr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= cfg.early_patience:
                    print(f"Early stopping. best_val_f1={best_val_f1:.4f}, best_thr={best_thr:.4f}")
                    break
        else:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_f1": best_val_f1,
        "best_thr": best_thr,
        "best_state_restored": best_state is not None,
    }


def knowable_batch_size(bs: int) -> int:
    # Keep as-is; helper for clarity if you later want different val batch size
    return bs


# -----------------------------
# 6) Example usage
# -----------------------------
if __name__ == "__main__":
    # Suppose you already have:
    #   train_list: List[Data]
    #   val_list: List[Data]
    #
    # Each Data.y is scalar with {0,1,2,3} (or {0,1}).
    #
    # Build datasets
    train_dataset = GraphDataset(train_list)
    val_dataset = GraphDataset(val_list)

    # Create model (set in_dim from your node feature dim)
    in_dim = train_list[0].x.size(-1)
    model = SageAttnBinary(in_dim=in_dim, hidden=128, num_layers=2, dropout=0.1)

    cfg = TrainConfig(
        batch_size=1,
        accum_steps=32,
        max_epochs=200,
        lr=1e-3,
        weight_decay=0.0,
        clip_grad_norm=1.0,
        gamma_pos=1.0,
        gamma_neg=4.0,
        clip=0.05,
        eval_every=1,
        early_patience=30,
        min_delta=1e-4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_asl_accum_oversample(model, train_dataset, val_dataset, device, cfg)
    print(result)