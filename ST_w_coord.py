import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Set Transformer building blocks
# -----------------------------
class MAB(nn.Module):
    """Multihead Attention Block: Q attends to K,V (like Transformer cross-attn)"""
    def __init__(self, dim, num_heads, ln=True, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln0 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim) if ln else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, Q, K, V, key_padding_mask=None, attn_mask=None):
        # Q,K,V: (B, Nq/Nk, D)
        H, _ = self.mha(Q, K, V, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        X = self.ln0(Q + H)
        X = self.ln1(X + self.ff(X))
        return X


class SAB(nn.Module):
    """Self-Attention Block"""
    def __init__(self, dim, num_heads, ln=True, dropout=0.0):
        super().__init__()
        self.mab = MAB(dim, num_heads, ln=ln, dropout=dropout)

    def forward(self, X, key_padding_mask=None):
        return self.mab(X, X, X, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    """Induced Set Attention Block"""
    def __init__(self, dim, num_heads, num_inducing, ln=True, dropout=0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)
        self.mab0 = MAB(dim, num_heads, ln=ln, dropout=dropout)  # I <- X
        self.mab1 = MAB(dim, num_heads, ln=ln, dropout=dropout)  # X <- I

    def forward(self, X, key_padding_mask=None):
        B = X.size(0)
        I = self.I.expand(B, -1, -1)  # (B, m, D)
        H = self.mab0(I, X, X, key_padding_mask=key_padding_mask)  # induced
        Y = self.mab1(X, H, H)  # X attends to induced
        return Y


# -----------------------------
# Coord-injected Set Transformer AutoEncoder
# -----------------------------
class CoordInjectedSetTransformerAE(nn.Module):
    """
    Input: tokens (B, N, Fin) + coords (B, N, 2)
    Output: recon tokens (B, N, Fout)
    Supports variable N (per batch padded) via key_padding_mask.
    """
    def __init__(
        self,
        fin: int,
        dim: int = 128,
        num_heads: int = 8,
        num_inducing: int = 32,
        enc_layers: int = 2,
        dec_layers: int = 2,
        coord_mode: str = "add",   # "add" or "concat"
        dropout: float = 0.0,
        fout: int = None,
    ):
        super().__init__()
        self.fin = fin
        self.fout = fin if fout is None else fout
        self.dim = dim
        assert coord_mode in ["add", "concat"]
        self.coord_mode = coord_mode

        # Feature embedding
        self.x_proj = nn.Linear(fin, dim)

        # Coord embedding
        self.c_proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # If concat, we fuse (x_emb || c_emb) -> dim
        self.fuse = nn.Linear(dim * 2, dim) if coord_mode == "concat" else nn.Identity()

        # Encoder: ISAB stack
        self.encoder = nn.ModuleList([
            ISAB(dim, num_heads, num_inducing, ln=True, dropout=dropout)
            for _ in range(enc_layers)
        ])

        # Bottleneck (optional)
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Decoder: cross-attn blocks
        # Query = (x_emb +/concat c_emb)  (i.e., token queries with coordinates)
        # Key/Value = latent set from encoder
        self.dec_blocks = nn.ModuleList([
            MAB(dim, num_heads, ln=True, dropout=dropout)
            for _ in range(dec_layers)
        ])

        # Final reconstruction head
        self.out = nn.Linear(dim, self.fout)

    def forward(self, x, coords, key_padding_mask=None):
        """
        x: (B, N, Fin)
        coords: (B, N, 2) in [0,1] recommended (normalized i/H, j/W)
        key_padding_mask: (B, N) bool, True for PAD tokens
        """
        # Embed
        x_emb = self.x_proj(x)             # (B, N, D)
        c_emb = self.c_proj(coords)        # (B, N, D)

        if self.coord_mode == "add":
            h = x_emb + c_emb
        else:  # concat
            h = self.fuse(torch.cat([x_emb, c_emb], dim=-1))

        # Encode (set -> set)
        for layer in self.encoder:
            h = layer(h, key_padding_mask=key_padding_mask)

        h = self.bottleneck(h)

        # Decode: token-queries attend to latent set (cross-attn)
        # q uses the same coordinate-injected token queries
        q = x_emb + c_emb if self.coord_mode == "add" else self.fuse(torch.cat([x_emb, c_emb], dim=-1))
        z = h

        y = q
        for block in self.dec_blocks:
            y = block(y, z, z, key_padding_mask=key_padding_mask)

        recon = self.out(y)  # (B, N, Fout)
        return recon


# -----------------------------
# Utility: build coords from (B,H,W) or (B,H,W,C)
# -----------------------------
def make_coords(B, H, W, device):
    """
    returns coords: (B, H*W, 2) normalized to [0,1]
    """
    ys = torch.linspace(0, 1, steps=H, device=device)
    xs = torch.linspace(0, 1, steps=W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
    c = torch.stack([yy, xx], dim=-1).view(1, H * W, 2)  # (1,N,2)
    return c.repeat(B, 1, 1)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example: scalar matrix input (B,H,W) -> tokens (B,N,1)
    B, H, W = 4, 10, 13
    x_mat = torch.randn(B, H, W, device=device)

    x = x_mat.view(B, H * W, 1)
    coords = make_coords(B, H, W, device)

    model = CoordInjectedSetTransformerAE(
        fin=1, dim=128, num_heads=8, num_inducing=32,
        enc_layers=2, dec_layers=2, coord_mode="add"
    ).to(device)

    recon = model(x, coords)  # (B, N, 1)

    # Per-cell reconstruction error -> anomaly score matrix (B,H,W)
    err = (recon - x).abs().view(B, H, W)
    print("err:", err.shape)