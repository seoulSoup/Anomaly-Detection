import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utils: masked stats / weak augment / InfoNCE
# ============================================================
def set_stats(x, mask=None):
    """
    x: [B, N, D]
    mask: [B, N] (True for valid)
    returns mean[B,D], std[B,D]
    """
    if mask is None:
        m = x.mean(dim=1)
        v = x.var(dim=1, unbiased=False).clamp_min(1e-12)
        return m, v.sqrt()
    else:
        msum = (x * mask.unsqueeze(-1)).sum(dim=1)
        count = mask.sum(dim=1).clamp_min(1)
        m = msum / count.unsqueeze(-1)
        diff2 = ((x - m.unsqueeze(1)) ** 2) * mask.unsqueeze(-1)
        v = diff2.sum(dim=1) / count.unsqueeze(-1)
        return m, v.sqrt()


def weak_augment(x, mask=None, jitter_std=0.02, drop_rate=0.1, scale_std=0.02, shift_std=0.02):
    """
    Mild augmentations that preserve identity of 'normal' sets while allowing contrastive learning.
    x: [B,N,D], mask: [B,N] or None (True for valid)
    """
    B, N, D = x.shape
    out = x.clone()

    # (1) feature jitter
    out = out + torch.randn_like(out) * jitter_std

    # (2) per-set affine (tiny scale + shift)
    scale = (1.0 + torch.randn(B, 1, 1, device=x.device) * scale_std).clamp(0.9, 1.1)
    shift = torch.randn(B, 1, 1, device=x.device) * shift_std
    out = out * scale + shift

    # (3) random dropout of elements (masking)
    if mask is None:
        m = torch.ones(B, N, device=x.device, dtype=torch.bool)
    else:
        m = mask.clone()

    if drop_rate > 0.0:
        keep = (torch.rand(B, N, device=x.device) > drop_rate)
        m = m & keep

    return out, m


def info_nce(z1, z2, temperature=0.2):
    """
    z1, z2: [B, H] matching pairs
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss


# ============================================================
# Set Transformer blocks (Lee et al., 2019)
# ============================================================
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, h=4):
        super().__init__()
        self.h = h
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.rff = nn.Sequential(
            nn.Linear(dim_V, dim_V * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim_V * 2, dim_V),
        )

    def forward(self, Q, K, mask_Q=None, mask_K=None):
        # Q: [B,Nq,Dq], K: [B,Nk,Dk]
        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)

        B, Nq, DV = Q_.shape
        Nk = K_.size(1)
        H = self.h
        assert DV % H == 0, f"dim_V={DV} must be divisible by heads={H}"
        d = DV // H

        def split_heads(x):
            return x.view(B, -1, H, d).transpose(1, 2)  # [B,H,N, d]

        Qh = split_heads(Q_)
        Kh = split_heads(K_)
        Vh = split_heads(V_)

        att = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d)  # [B,H,Nq,Nk]

        if mask_K is not None:
            # mask_K: [B,Nk] True=valid
            maskK = mask_K.unsqueeze(1).unsqueeze(2)  # [B,1,1,Nk]
            att = att.masked_fill(~maskK, float("-inf"))

        A = att.softmax(dim=-1)  # [B,H,Nq,Nk]
        Hout = A @ Vh  # [B,H,Nq,d]
        Hout = Hout.transpose(1, 2).contiguous().view(B, Nq, DV)

        Hout = self.ln0(Hout + Q_)
        Hout2 = self.ln1(self.rff(Hout) + Hout)
        return Hout2


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, h=4):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, h)

    def forward(self, X, mask=None):
        return self.mab(X, X, mask_Q=mask, mask_K=mask)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, m=64, h=4):
        super().__init__()
        self.I = nn.Parameter(torch.randn(m, dim_out) * 0.02)
        self.mab0 = MAB(dim_out, dim_in, dim_out, h)
        self.mab1 = MAB(dim_in, dim_out, dim_out, h)

    def forward(self, X, mask=None):
        B = X.size(0)
        I = self.I.unsqueeze(0).expand(B, -1, -1)  # [B,m,dim_out]
        H = self.mab0(I, X, mask_K=mask)           # [B,m,dim_out]
        return self.mab1(X, H, mask_Q=mask)        # [B,N,dim_out]


class PMA(nn.Module):
    def __init__(self, dim, k=1, h=4):
        super().__init__()
        self.S = nn.Parameter(torch.randn(k, dim) * 0.02)
        self.mab = MAB(dim, dim, dim, h)

    def forward(self, X, mask=None):
        B = X.size(0)
        S = self.S.unsqueeze(0).expand(B, -1, -1)  # [B,k,dim]
        return self.mab(S, X, mask_K=mask)         # [B,k,dim]


# ============================================================
# Positional embedding for 2D coords
# ============================================================
class PosEmbed2D(nn.Module):
    """
    coords: [B,N,2] in [0,1] or [-1,1]
    returns: [B,N,d_hid]
    """
    def __init__(self, d_hid=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_hid),
        )

    def forward(self, coords):
        return self.mlp(coords)


# ============================================================
# Model: Encoder / Decoder / StatHead / OneClass wrapper
# ============================================================
class SetEncoder(nn.Module):
    def __init__(
        self,
        d_in=1,
        d_hid=128,
        n_isab=2,
        m_induce=64,
        heads=4,
        pos_mode="add",  # "add" | "concat" | "none"
        pos_alpha=1.0,
    ):
        super().__init__()
        assert pos_mode in ["add", "concat", "none"]
        self.pos_mode = pos_mode
        self.pos_alpha = pos_alpha

        self.proj = nn.Linear(d_in, d_hid)
        self.pos = PosEmbed2D(d_hid) if pos_mode != "none" else None
        self.fuse = nn.Linear(d_hid * 2, d_hid) if pos_mode == "concat" else None

        self.blocks = nn.ModuleList([ISAB(d_hid, d_hid, m=m_induce, h=heads) for _ in range(n_isab)])
        self.pma = PMA(d_hid, k=1, h=heads)

    def forward(self, X, mask=None, coords=None):
        Xh = self.proj(X)  # [B,N,H]

        if self.pos_mode != "none":
            if coords is None:
                raise ValueError("coords must be provided when pos_mode != 'none'")
            Ph = self.pos(coords)  # [B,N,H]
            if self.pos_mode == "add":
                Xh = Xh + self.pos_alpha * Ph
            else:  # concat
                Xh = self.fuse(torch.cat([Xh, Ph], dim=-1))

        for blk in self.blocks:
            Xh = blk(Xh, mask=mask)  # [B,N,H]

        Z_set = self.pma(Xh, mask=mask)[:, 0, :]  # [B,H]
        return Xh, Z_set


class ElemDecoder(nn.Module):
    def __init__(self, d_hid=128, d_out=1):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(inplace=True),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, Z_elems):
        return self.dec(Z_elems)  # [B,N,d_out]


class StatHead(nn.Module):
    """Predict set mean and std from set embedding."""
    def __init__(self, d_hid=128, d_out=2):  # (mu, sigma)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.ReLU(inplace=True),
            nn.Linear(d_hid, d_out),
        )

    def forward(self, z_set):
        out = self.mlp(z_set)  # [B,2]
        mu_hat = out[:, :1]
        sigma_hat = F.softplus(out[:, 1:2]) + 1e-6
        return mu_hat, sigma_hat


class OneClassSetModel(nn.Module):
    def __init__(
        self,
        d_in=1,
        d_hid=128,
        n_isab=2,
        m_induce=64,
        heads=4,
        pos_mode="add",
        pos_alpha=1.0,
    ):
        super().__init__()
        self.encoder = SetEncoder(d_in, d_hid, n_isab, m_induce, heads, pos_mode=pos_mode, pos_alpha=pos_alpha)
        self.decoder = ElemDecoder(d_hid, d_in)
        self.stat_head = StatHead(d_hid, 2)
        self.register_buffer("center", torch.zeros(d_hid))

    def forward(self, X, mask=None, coords=None):
        Z_elems, Z_set = self.encoder(X, mask=mask, coords=coords)
        X_hat = self.decoder(Z_elems)
        mu_hat, sigma_hat = self.stat_head(Z_set)
        return X_hat, Z_elems, Z_set, mu_hat, sigma_hat


# ============================================================
# Collate: variable HxW matrices -> padded tokens + mask + coords
# ============================================================
def _to_hw_c(x):
    """
    Normalize sample to shape [H, W, C]
    Accepts:
      - [H, W] (scalar)
      - [H, W, C]
    """
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected [H,W] or [H,W,C], got {tuple(x.shape)}")


def _make_coords_hw(H: int, W: int, device, normalize="01"):
    """
    coords: [H*W, 2], normalized.
    """
    ys = torch.linspace(0, 1, steps=H, device=device)
    xs = torch.linspace(0, 1, steps=W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
    c = torch.stack([yy, xx], dim=-1).view(H * W, 2)  # [N,2]
    if normalize == "11":
        c = c * 2 - 1
    return c


def collate_var_hw(batch: List[Dict[str, Any]], coord_norm="01") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    batch item format (dict):
      {
        "x": Tensor [H,W] or [H,W,C],
        ... (optional meta)
      }

    returns:
      X      [B, Nmax, C]
      mask   [B, Nmax] (True=valid)
      coords [B, Nmax, 2]
      meta   dict with per-sample (H,W,N)
    """
    # Find device later (usually CPU in collate; move to device in training step)
    xs = [ _to_hw_c(item["x"]) for item in batch ]
    Hs = [ t.size(0) for t in xs ]
    Ws = [ t.size(1) for t in xs ]
    Cs = [ t.size(2) for t in xs ]
    if len(set(Cs)) != 1:
        raise ValueError(f"All samples must share same C, got {Cs}")
    C = Cs[0]

    Ns = [ h * w for h, w in zip(Hs, Ws) ]
    Nmax = max(Ns)
    B = len(xs)

    X = torch.zeros(B, Nmax, C, dtype=xs[0].dtype)              # padded tokens
    mask = torch.zeros(B, Nmax, dtype=torch.bool)               # True valid
    coords = torch.zeros(B, Nmax, 2, dtype=torch.float32)       # padded coords

    for b, (x_hw_c, H, W, N) in enumerate(zip(xs, Hs, Ws, Ns)):
        # tokens
        tok = x_hw_c.reshape(N, C)
        X[b, :N] = tok
        mask[b, :N] = True

        # coords
        c = _make_coords_hw(H, W, device=coords.device, normalize=coord_norm)  # [N,2]
        coords[b, :N] = c

    meta = {"H": Hs, "W": Ws, "N": Ns, "Nmax": Nmax, "C": C}
    return X, mask, coords, meta


# ============================================================
# Example training step (recon + optional InfoNCE)
# ============================================================
@dataclass
class LossWeights:
    recon: float = 1.0
    stats: float = 0.0
    nce: float = 0.0


def masked_l1(x_hat, x, mask):
    # x_hat,x: [B,N,C], mask: [B,N] True valid
    diff = (x_hat - x).abs()
    diff = diff * mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1).to(diff.dtype)
    return diff.sum() / denom


def train_step(model, batch, device, lw: LossWeights, temperature=0.2):
    X, mask, coords, meta = batch
    X = X.to(device)
    mask = mask.to(device)
    coords = coords.to(device)

    # view1/view2 for NCE (optional)
    if lw.nce > 0.0:
        X1, m1 = weak_augment(X, mask=mask)
        X2, m2 = weak_augment(X, mask=mask)

        Xhat1, Ze1, Zs1, mu1, sig1 = model(X1, mask=m1, coords=coords)
        Xhat2, Ze2, Zs2, mu2, sig2 = model(X2, mask=m2, coords=coords)

        loss_recon = 0.5 * (masked_l1(Xhat1, X1, m1) + masked_l1(Xhat2, X2, m2))
        loss_nce = info_nce(Zs1, Zs2, temperature=temperature)
        loss = lw.recon * loss_recon + lw.nce * loss_nce
        logs = {"loss": loss.item(), "recon": loss_recon.item(), "nce": loss_nce.item()}
        return loss, logs

    # recon only
    Xhat, Ze, Zs, mu_hat, sigma_hat = model(X, mask=mask, coords=coords)
    loss_recon = masked_l1(Xhat, X, mask)
    loss = lw.recon * loss_recon
    logs = {"loss": loss.item(), "recon": loss_recon.item()}
    return loss, logs


# ============================================================
# Quick sanity check
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fake batch: variable H,W
    batch_list = [
        {"x": torch.randn(10, 13)},      # [H,W]
        {"x": torch.randn(7, 9)},        # [H,W]
        {"x": torch.randn(12, 5)},       # [H,W]
    ]

    X, mask, coords, meta = collate_var_hw(batch_list, coord_norm="01")
    print("X:", X.shape, "mask:", mask.shape, "coords:", coords.shape, "meta:", meta)

    model = OneClassSetModel(
        d_in=1, d_hid=128, n_isab=2, m_induce=64, heads=4,
        pos_mode="add", pos_alpha=0.7
    ).to(device)

    # Ensure X has shape [B,N,C]
    if X.dim() == 3 and X.size(-1) == 1:
        pass
    else:
        raise RuntimeError("Expected scalar feature => C=1 in this example")

    lw = LossWeights(recon=1.0, nce=0.2)
    loss, logs = train_step(model, (X, mask, coords, meta), device, lw, temperature=0.2)
    print(logs)

    # Anomaly score matrix per-sample로 다시 HxW로 되돌리고 싶으면:
    # score_tok = (Xhat - X).abs()  # [B,N,1]
    # 각 샘플의 N = H*W 만큼만 잘라서 [H,W]로 reshape하면 됨.