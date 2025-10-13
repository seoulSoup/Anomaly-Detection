import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 안전한 어텐션 유틸
# ---------------------------
def safe_softmax(logits, mask=None, dim=-1, eps=1e-12):
    # logits: [..., K]
    if mask is not None:
        logits = logits.masked_fill(~mask, float('-inf'))
        all_masked = (~mask).all(dim=dim, keepdim=True)  # 행 전체가 가려진 경우
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
    logits = logits.clamp(-30.0, 30.0)  # overflow 방지
    attn = F.softmax(logits, dim=dim)
    if mask is not None:
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)
    denom = attn.sum(dim=dim, keepdim=True).clamp_min(eps)  # 분모 0 방지
    return attn / denom

# ---------------------------
# Set Transformer 구성요소
# ---------------------------
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_KV, dim_out, num_heads=4, ln=True):
        super().__init__()
        self.h = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_out, bias=False)
        self.fc_k = nn.Linear(dim_KV, dim_out, bias=False)
        self.fc_v = nn.Linear(dim_KV, dim_out, bias=False)
        self.fc_o = nn.Linear(dim_out, dim_out, bias=False)
        self.ln0 = nn.LayerNorm(dim_out, eps=1e-4) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_out, eps=1e-4) if ln else nn.Identity()
        self.ff = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, Q, K, mask_Q=None, mask_K=None):
        # Q: [B,Nq,Dq], K: [B,Nk,Dk]
        B, Nq, _ = Q.shape
        Nk = K.size(1)
        D = self.fc_o.out_features
        H = self.h
        Dh = D // H

        Qh = self.fc_q(Q).float().view(B, Nq, H, Dh).transpose(1, 2)  # [B,H,Nq,Dh]
        Kh = self.fc_k(K).float().view(B, Nk, H, Dh).transpose(1, 2)  # [B,H,Nk,Dh]
        Vh = self.fc_v(K).float().view(B, Nk, H, Dh).transpose(1, 2)  # [B,H,Nk,Dh]

        scale = Dh ** -0.5
        logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale  # [B,H,Nq,Nk]

        # 마스크 구성 (True=유효)
        if mask_K is not None:
            maskK = mask_K[:, None, None, :].expand(B, H, Nq, Nk)
        else:
            maskK = torch.ones(B, H, Nq, Nk, dtype=torch.bool, device=Q.device)

        attn = safe_softmax(logits, mask=maskK, dim=-1)  # [B,H,Nq,Nk]
        X = torch.matmul(attn, Vh)  # [B,H,Nq,Dh]
        X = X.transpose(1, 2).contiguous().view(B, Nq, D)

        # 잔차/정규화/FF
        H0 = self.ln0(self.fc_o(X) + self.fc_q(Q))  # Q 경로와 차원 맞춤
        H1 = self.ln1(H0 + self.ff(H0))
        return H1

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, ln=True):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads=num_heads, ln=ln)
    def forward(self, X, mask=None):
        return self.mab(X, X, mask_Q=mask, mask_K=mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads=4, num_induce=16, ln=True):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_induce, dim_out) * 0.01)  # 작은 초기화
        self.mab1 = MAB(dim_out, dim_in, dim_out, num_heads=num_heads, ln=ln)  # Q=I, K=X
        self.mab2 = SAB(dim_out, dim_out, num_heads=num_heads, ln=ln)          # Q=H, K=H
    def forward(self, X, mask=None):
        B = X.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, mask_Q=None, mask_K=mask)  # [B,M,D]
        return self.mab2(H, mask=None)                 # [B,M,D]

class PMA(nn.Module):
    def __init__(self, dim, num_seeds=1, num_heads=4, ln=True):
        super().__init__()
        self.S = nn.Parameter(torch.zeros(1, num_seeds, dim))
        nn.init.xavier_normal_(self.S, gain=0.01)  # 작은 분산
        self.mab = MAB(dim, dim, dim, num_heads=num_heads, ln=ln)
    def forward(self, X, mask=None):
        B = X.size(0)
        S = self.S.expand(B, -1, -1).to(dtype=X.dtype)
        return self.mab(S, X, mask_Q=None, mask_K=mask)  # [B,S,D]

# ---------------------------
# MAE-Style Set Autoencoder
# ---------------------------
class SetMAE(nn.Module):
    """
    입력: X=[B,N,1], valid_mask=[B,N] (True=유효)
    MAE 마스킹: mae_mask=[B,N] (True=가린 토큰; 복원 대상)
    인코더 입력 피처: [value_or_0, is_masked] → dim_in=2
    """
    def __init__(self, dim_model=64, num_heads=4, num_induce=16, num_isab=2, pma_seeds=1):
        super().__init__()
        dim_in = 2   # (값, 마스크플래그)
        D = dim_model
        self.in_proj = nn.Linear(dim_in, D)

        self.isab_blocks = nn.ModuleList([
            ISAB(D, D, num_heads=num_heads, num_induce=num_induce) for _ in range(num_isab)
        ])
        # 토큰 컨텍스트를 seed로 요약 (선택적): 복원 시 토큰+세트표현 결합에 사용
        self.pma = PMA(D, num_seeds=pma_seeds, num_heads=num_heads)
        # 토큰별 복원 헤드: [토큰 임베딩 + 세트 요약] -> 스칼라
        self.dec = nn.Sequential(
            nn.Linear(D + D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

    def forward(self, X, valid_mask, mae_mask):
        """
        X: [B,N,1], valid_mask: [B,N] (True=유효), mae_mask: [B,N] (True=가릴 토큰)
        """
        B, N, _ = X.shape
        device = X.device

        # 1) MAE 입력 만들기: 가릴 위치는 값을 0으로, 대신 "is_masked" 채널을 1로
        is_masked_feat = mae_mask.float().unsqueeze(-1)           # [B,N,1]
        x_masked = X.masked_fill(mae_mask.unsqueeze(-1), 0.0)     # [B,N,1]
        enc_in = torch.cat([x_masked, is_masked_feat], dim=-1)    # [B,N,2]

        # 2) 인코딩
        H = self.in_proj(enc_in)                                   # [B,N,D]
        for blk in self.isab_blocks:
            H = blk(H, mask=valid_mask)                            # [B,M,D] (ISAB 출력은 M개 유도 토큰)
        # ISAB의 출력은 inducing 토큰 수 M 기준. 복원을 토큰 단위로 하려면
        # 원 토큰 컨텍스트를 H_token으로 유지하는 대안도 있지만,
        # 여기서는 세트 요약을 붙여 각 원 토큰을 복원하도록 설계:
        Z_set = self.pma(H, mask=None)                             # [B,S,D]
        Z_set = Z_set.mean(dim=1)                                  # [B,D] (seed 여러 개면 평균)

        # 3) 토큰별 복원: 각 원 토큰 입력(가려진 값 포함)과 세트 요약을 결합
        #    원 토큰 위치별 임베딩이 필요하므로, 간단히 입력을 다시 임베딩해 토큰별 특징으로 사용
        #    (선택) 더 정교하게 하려면 별도 Token-SAB로 X_token→H_token 만든 뒤 concat
        token_feat = self.in_proj(enc_in)                          # [B,N,D]
        Z_set_exp = Z_set.unsqueeze(1).expand(B, N, Z_set.size(-1))# [B,N,D]
        dec_in = torch.cat([token_feat, Z_set_exp], dim=-1)        # [B,N,2D]
        X_hat = self.dec(dec_in)                                   # [B,N,1]

        return X_hat

# ---------------------------
# 마스킹/손실/학습 루프
# ---------------------------
def make_mae_mask(valid_mask, mask_ratio=0.5):
    """
    valid 중에서 mask_ratio만큼 True로 선택 (나머지는 False)
    """
    B, N = valid_mask.shape
    device = valid_mask.device
    rand = torch.rand(B, N, device=device)
    # 유효한 위치만 랜덤값, 무효는 +inf로 만들어 선택되지 않게
    rand = torch.where(valid_mask, rand, torch.ones_like(rand) * 2.0)
    k = (valid_mask.sum(dim=1).float() * mask_ratio).clamp_min(1).long()  # 최소 1개는 가리기
    mae_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        vals, idx = torch.topk(rand[b] * -1, k[b], largest=True)  # 작은 값 우선 선택
        mae_mask[b, idx] = True
    return mae_mask

def mae_recon_loss(x_hat, x, valid_mask, mae_mask):
    """
    마스킹된 + 유효한 위치에만 MSE
    """
    target_mask = valid_mask & mae_mask                           # [B,N]
    if target_mask.sum() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    diff = (x_hat - x).squeeze(-1)                                # [B,N]
    loss = (diff[target_mask] ** 2).mean()
    return loss

# ---------------------------
# 사용 예시
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    # 가짜 배치: B=8, N(max)=20, 1D 값
    B, N = 8, 20
    X = torch.randn(B, N, 1, device=device)
    # 가변 길이 흉내: 각 샘플마다 유효 길이 L_b
    lengths = torch.randint(low=8, high=N+1, size=(B,))
    valid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b, L in enumerate(lengths):
        valid_mask[b, :L] = True

    model = SetMAE(dim_model=64, num_heads=4, num_induce=16, num_isab=2, pma_seeds=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(1, 11):
        mae_mask = make_mae_mask(valid_mask, mask_ratio=0.5)      # 마스킹 비율 50%
        X_hat = model(X, valid_mask, mae_mask)
        loss = mae_recon_loss(X_hat, X, valid_mask, mae_mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        print(f"[Epoch {epoch:02d}] MAE recon loss (masked only): {loss.item():.6f}")