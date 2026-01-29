# train_local_pattern_gnn.py
# ------------------------------------------------------------
# Local-pattern binary classifier using kNN graph + edge-aware MPNN (PyG)
# Input per sample: L x 3 (score, x, y), x,y in [0,1], L is variable.
# Output: pass/fail (binary classification) per sample/graph.
# ------------------------------------------------------------

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# PyTorch Geometric / torch_scatter
from torch_geometric.nn import knn_graph
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism tradeoff: can slow down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Model components
# -------------------------
class EdgeMP(nn.Module):
    """
    Edge-conditioned message passing:
      m_ij = MLP([h_i(dst), h_j(src), e_ij])
      agg_i = sum_{j in N(i)} m_ij
      h_i' = h_i + MLP([h_i, agg_i])
    """
    def __init__(self, d_model: int, d_edge: int, d_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * d_model + d_edge, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )
        self.norm = GraphNorm(d_model)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor):
        src, dst = edge_index  # src -> dst
        m_in = torch.cat([h[dst], h[src], edge_attr], dim=-1)
        m = self.msg_mlp(m_in)

        agg = scatter_add(m, dst, dim=0, dim_size=h.size(0))
        upd_in = torch.cat([h, agg], dim=-1)
        dh = self.upd_mlp(upd_in)

        h = h + F.dropout(dh, p=self.dropout, training=self.training)
        h = self.norm(h, batch)
        return h


class GatedAttnPool(nn.Module):
    """
    Graph-wise gated attention pooling:
      a_i = softmax(g(h_i)) within each graph
      g = sum_i a_i * h_i
    """
    def __init__(self, d_model: int, d_hidden: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor):
        logits = self.gate(h).squeeze(-1)     # [N]
        a = softmax(logits, batch)            # [N] graph-wise
        pooled = scatter_add(h * a.unsqueeze(-1), batch, dim=0)  # [B, d_model]
        return pooled


class LocalPatternBinaryClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        k: int = 12,
        num_layers: int = 3,
        dropout: float = 0.1,
        score_compress: str = "none",  # none | tanh | log1p
        score_scale: float = 5.0,      # for tanh compression
    ):
        super().__init__()
        self.k = k
        self.score_compress = score_compress
        self.score_scale = score_scale

        self.in_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # edge_attr dims: dx, dy, dist, ds  => 4
        d_edge = 4
        self.layers = nn.ModuleList([
            EdgeMP(d_model=d_model, d_edge=d_edge, d_hidden=2 * d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pool = GatedAttnPool(d_model, d_hidden=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),  # logits
        )

    def _compress_score(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,3]
        if self.score_compress == "none":
            return x
        s = x[:, 0:1]
        if self.score_compress == "tanh":
            s2 = torch.tanh(s / max(self.score_scale, 1e-6))
        elif self.score_compress == "log1p":
            s2 = torch.sign(s) * torch.log1p(torch.abs(s))
        else:
            raise ValueError(f"Unknown score_compress: {self.score_compress}")
        return torch.cat([s2, x[:, 1:3]], dim=-1)

    @staticmethod
    def build_edge_attr(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N,3] = [score, x, y]
        edge_index: [2,E] src->dst
        edge_attr: [E,4] = [dx, dy, dist, ds]
        """
        src, dst = edge_index
        s_src = x[src, 0:1]
        s_dst = x[dst, 0:1]
        p_src = x[src, 1:3]
        p_dst = x[dst, 1:3]

        dp = p_src - p_dst  # [E,2]
        dist = torch.sqrt((dp ** 2).sum(dim=-1, keepdim=True) + 1e-12)  # [E,1]
        ds = s_src - s_dst  # [E,1]
        return torch.cat([dp, dist, ds], dim=-1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x: [N,3], batch: [N] graph id
        returns logits: [B]
        """
        x = self._compress_score(x)
        pos = x[:, 1:3]

        # kNN graph on coordinates; batch separates graphs
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=False)
        edge_attr = self.build_edge_attr(x, edge_index)

        h = self.in_mlp(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, batch)

        g = self.pool(h, batch)                  # [B, d_model]
        logits = self.head(g).squeeze(-1)        # [B]
        return logits


# -------------------------
# Dataset / Collate
# -------------------------
@dataclass
class Sample:
    x: torch.Tensor   # [L,3]
    y: int           # 0/1


class InMemorySetDataset(Dataset):
    """
    Replace this with your actual dataset loader.
    Expected: list of samples, each is (L,3) float tensor and y in {0,1}.
    """
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def collate_variable_L(batch: List[Sample]) -> Dict[str, Any]:
    """
    Converts a list of variable-length (L,3) into a packed representation:
      x_cat: [N,3] where N=sum L
      batch_vec: [N] graph id per node
      y: [B] labels
      lengths: [B]
    """
    xs = []
    bs = []
    ys = []
    lengths = []
    for b, s in enumerate(batch):
        x = s.x
        if x.dim() != 2 or x.size(-1) != 3:
            raise ValueError(f"Each sample x must be [L,3], got {tuple(x.shape)}")
        L = x.size(0)
        xs.append(x)
        bs.append(torch.full((L,), b, dtype=torch.long))
        ys.append(int(s.y))
        lengths.append(L)

    x_cat = torch.cat(xs, dim=0)                       # [N,3]
    batch_vec = torch.cat(bs, dim=0)                   # [N]
    y = torch.tensor(ys, dtype=torch.float32)          # [B]
    lengths = torch.tensor(lengths, dtype=torch.long)  # [B]
    return {"x": x_cat, "batch": batch_vec, "y": y, "lengths": lengths}


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    logits: [B], y: [B] float {0,1}
    returns: accuracy, precision, recall, f1, auc (if possible)
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()

    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # AUC (optional, needs sklearn)
    auc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        # if only one class present, roc_auc_score fails
        if (y.min().item() != y.max().item()):
            auc = float(roc_auc_score(y.cpu().numpy(), probs.cpu().numpy()))
    except Exception:
        pass

    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[float] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_logits = []
    all_y = []

    for batch in loader:
        x = batch["x"].to(device)
        bvec = batch["batch"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, bvec)  # [B]
        loss = criterion(logits, y)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        all_logits.append(logits.detach())
        all_y.append(y.detach())

    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    metrics = compute_metrics_from_logits(logits_cat, y_cat)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: Optional[float] = None,
) -> Dict[str, float]:
    model.eval()
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_logits = []
    all_y = []

    for batch in loader:
        x = batch["x"].to(device)
        bvec = batch["batch"].to(device)
        y = batch["y"].to(device)

        logits = model(x, bvec)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        all_logits.append(logits)
        all_y.append(y)

    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    metrics = compute_metrics_from_logits(logits_cat, y_cat)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


# -------------------------
# Example synthetic data (for sanity check)
# Replace with your real loader.
# -------------------------
def make_synthetic_samples(n: int, L_min: int, L_max: int, seed: int = 0) -> List[Sample]:
    """
    Generates a toy problem:
      - Points distributed in [0,1]^2
      - label=1 if there exists a local "hot cluster" where score is high in a tight area
      This is just to test the pipeline runs end-to-end.
    """
    g = torch.Generator().manual_seed(seed)
    samples = []

    for i in range(n):
        L = int(torch.randint(L_min, L_max + 1, (1,), generator=g).item())
        xy = torch.rand((L, 2), generator=g)

        # base score noise
        score = 0.2 * torch.randn((L, 1), generator=g)

        # inject local anomaly cluster in some samples
        y = 1 if torch.rand((), generator=g).item() < 0.5 else 0
        if y == 1 and L >= 10:
            center = torch.rand((1, 2), generator=g)
            dist = torch.sqrt(((xy - center) ** 2).sum(dim=-1, keepdim=True) + 1e-12)
            bump = torch.exp(- (dist ** 2) / (2 * (0.06 ** 2)))  # local bump
            score = score + 3.0 * bump  # high local score

        x = torch.cat([score, xy], dim=-1).float()
        samples.append(Sample(x=x, y=y))

    return samples


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--score_compress", type=str, default="none", choices=["none", "tanh", "log1p"])
    parser.add_argument("--score_scale", type=float, default=5.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="best.pt")

    # synthetic data options (replace with real)
    parser.add_argument("--use_synth", action="store_true")
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=400)
    parser.add_argument("--L_min", type=int, default=30)
    parser.add_argument("--L_max", type=int, default=120)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    # -------------------------
    # Load dataset
    # -------------------------
    if args.use_synth:
        train_samples = make_synthetic_samples(args.n_train, args.L_min, args.L_max, seed=args.seed + 1)
        val_samples = make_synthetic_samples(args.n_val, args.L_min, args.L_max, seed=args.seed + 2)
        test_samples = make_synthetic_samples(args.n_test, args.L_min, args.L_max, seed=args.seed + 3)
    else:
        # TODO: Replace this block with your real loading code.
        # You should produce: train_samples/val_samples/test_samples as List[Sample],
        # where Sample.x is a torch.FloatTensor [L,3] and Sample.y is 0/1 int.
        raise RuntimeError(
            "Real dataset loader not implemented. "
            "Run with --use_synth first, then replace the dataset loading block."
        )

    train_ds = InMemorySetDataset(train_samples)
    val_ds = InMemorySetDataset(val_samples)
    test_ds = InMemorySetDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_variable_L)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_variable_L)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_variable_L)

    # class imbalance handling (pos_weight = #neg/#pos)
    y_train = torch.tensor([s.y for s in train_samples], dtype=torch.float32)
    n_pos = float(y_train.sum().item())
    n_neg = float((1 - y_train).sum().item())
    pos_weight = (n_neg / max(n_pos, 1.0)) if (n_pos > 0 and n_neg > 0) else None

    model = LocalPatternBinaryClassifier(
        d_model=args.d_model,
        k=args.k,
        num_layers=args.num_layers,
        dropout=args.dropout,
        score_compress=args.score_compress,
        score_scale=args.score_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_path = args.save_path

    print(f"Device: {device}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"pos_weight: {pos_weight}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device=device,
            pos_weight=pos_weight, grad_clip=args.grad_clip
        )
        val_metrics = evaluate(model, val_loader, device=device, pos_weight=pos_weight)
        dt = time.time() - t0

        line = (
            f"[{epoch:03d}/{args.epochs}] {dt:.1f}s | "
            f"train loss {train_metrics['loss']:.4f} f1 {train_metrics['f1']:.4f} acc {train_metrics['acc']:.4f} auc {train_metrics['auc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} f1 {val_metrics['f1']:.4f} acc {val_metrics['acc']:.4f} auc {val_metrics['auc']:.4f}"
        )
        print(line)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    print(f"Best val f1: {best_val_f1:.4f} | saved to {best_path}")

    # Test using best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device=device, pos_weight=pos_weight)
    print(
        f"[TEST] loss {test_metrics['loss']:.4f} "
        f"f1 {test_metrics['f1']:.4f} acc {test_metrics['acc']:.4f} "
        f"precision {test_metrics['precision']:.4f} recall {test_metrics['recall']:.4f} auc {test_metrics['auc']:.4f}"
    )


if __name__ == "__main__":
    main()