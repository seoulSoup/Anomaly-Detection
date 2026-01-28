import torch
import torch.nn as nn
import torch.nn.functional as F

class NegLogitMemory:
    """
    Fixed-size memory queue for negative logits.
    Stores detached logits (CPU/GPU 상관 없이 Tensor로 저장 가능).
    """
    def __init__(self, capacity: int = 8192, device: str | torch.device = "cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.buf = torch.empty((0,), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def push_neg(self, neg_logits: torch.Tensor):
        """
        neg_logits: (K,) tensor of logits for negative samples (y=0). Detached/undetched OK.
        """
        if neg_logits.numel() == 0:
            return
        x = neg_logits.detach().view(-1).to(self.device, dtype=torch.float32)
        self.buf = torch.cat([self.buf, x], dim=0)
        if self.buf.numel() > self.capacity:
            self.buf = self.buf[-self.capacity:]  # keep most recent

    @torch.no_grad()
    def sample_hard_negs(self, k: int = 128) -> torch.Tensor:
        """
        Returns top-k largest negative logits from memory (hard negatives).
        If buffer < k, returns all.
        """
        if self.buf.numel() == 0:
            return self.buf
        k = min(int(k), self.buf.numel())
        # hard negatives = largest logits (most positive => most confusing)
        return torch.topk(self.buf, k=k, largest=True).values

def memory_ranking_loss(
    pos_logits: torch.Tensor,
    neg_memory_logits: torch.Tensor,
    margin: float = 1.0,
    mode: str = "hinge",   # "hinge" or "softplus"
) -> torch.Tensor:
    """
    pos_logits: (P,) logits from positive samples in current step
    neg_memory_logits: (K,) sampled hard negative logits from memory
    Want: pos_logits >= neg_logits + margin

    Returns scalar loss.
    """
    if pos_logits.numel() == 0 or neg_memory_logits.numel() == 0:
        return pos_logits.new_tensor(0.0)

    pos = pos_logits.view(-1, 1)   # (P,1)
    neg = neg_memory_logits.view(1, -1).to(pos.device, pos.dtype)  # (1,K)

    # diff = pos - neg  ; want diff >= margin
    diff = pos - neg

    if mode == "hinge":
        # relu(margin - diff)
        return F.relu(margin - diff).mean()
    elif mode == "softplus":
        # softplus(margin - diff) = log(1 + exp(margin - diff))
        return F.softplus(margin - diff).mean()
    else:
        raise ValueError("mode must be 'hinge' or 'softplus'")
        
@torch.no_grad()
def average_precision_torch(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    scores: (N,) higher = more positive (sigmoid(prob) or raw logits도 가능)
    labels: (N,) 0/1
    Computes Average Precision (AUPRC) in a standard way using sorting.
    """
    scores = scores.view(-1).float()
    labels = labels.view(-1).long()

    P = int(labels.sum().item())
    N = labels.numel()
    if N == 0 or P == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    y = labels[order]

    tp = torch.cumsum(y, dim=0).float()
    fp = torch.cumsum(1 - y, dim=0).float()

    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / float(P)

    # AP = sum over points where y==1 of precision at that point / P
    ap = (precision[y == 1].sum() / float(P)).item()
    return float(ap)
    
def train_one_epoch_with_memory(
    model: nn.Module,
    loader,
    optimizer,
    asl_criterion: nn.Module,
    neg_mem: NegLogitMemory,
    device: torch.device,
    accum_steps: int = 32,
    clip_grad_norm: float = 1.0,
    # ranking settings
    rank_lambda: float = 0.5,
    rank_margin: float = 1.0,
    hard_k: int = 128,
    rank_mode: str = "hinge",  # or "softplus"
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    steps = 0
    global_step = 0

    for batch in loader:
        steps += 1
        global_step += 1
        batch = batch.to(device)

        logits = model(batch).view(-1)              # (num_graphs,)
        y_bin = (batch.y.view(-1) != 0).float()     # (num_graphs,) float 0/1

        # ---- ASL loss (classification-like) ----
        loss_asl = asl_criterion(logits, y_bin)

        # ---- Memory ranking loss (ranking-like) ----
        pos_logits = logits[y_bin == 1]
        hard_negs = neg_mem.sample_hard_negs(k=hard_k)  # from memory (detached)
        loss_rank = memory_ranking_loss(pos_logits, hard_negs, margin=rank_margin, mode=rank_mode)

        loss = loss_asl + rank_lambda * loss_rank

        # grad accumulation
        (loss / accum_steps).backward()
        total_loss += float(loss.item())

        # Update memory with current step's negative logits (detached)
        with torch.no_grad():
            neg_logits_cur = logits[y_bin == 0]
            neg_mem.push_neg(neg_logits_cur)

        if (global_step % accum_steps) == 0:
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # flush leftover
    if (global_step % accum_steps) != 0:
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, steps)
    
@torch.no_grad()
def eval_ap_and_bestf1(model, loader, device):
    model.eval()
    all_logits = []
    all_y = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).view(-1)
        y = (batch.y.view(-1) != 0).long()
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)

    # AUPRC는 logits 그대로 써도 ranking은 동일하지만,
    # 보기 편하게 sigmoid(scores)로 해도 됨.
    scores = torch.sigmoid(logits)

    ap = average_precision_torch(scores, y)

    # (선택) best F1 threshold (정렬 기반) — 빠름
    best_f1, best_thr = best_f1_threshold_fast(scores, y)

    # 분포 모니터링
    pos = y == 1
    neg = y == 0
    pos_mean = float(scores[pos].mean().item()) if pos.any() else float("nan")
    neg_mean = float(scores[neg].mean().item()) if neg.any() else float("nan")
    pos_max  = float(scores[pos].max().item())  if pos.any() else float("nan")
    neg_max  = float(scores[neg].max().item())  if neg.any() else float("nan")

    return {
        "ap": ap,
        "best_f1": float(best_f1),
        "best_thr": float(best_thr),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pos_max": pos_max,
        "neg_max": neg_max,
        "pos_cnt": int(pos.sum().item()),
        "neg_cnt": int(neg.sum().item()),
    }
    
def train_loop_score_ranking(
    model,
    train_loader,
    val_loader,
    optimizer,
    asl_criterion,
    device,
    # memory
    mem_capacity: int = 8192,
    mem_device: str = "cpu",
    # train
    max_epochs: int = 200,
    accum_steps: int = 32,
    clip_grad_norm: float = 1.0,
    # ranking
    rank_lambda: float = 0.5,
    rank_margin: float = 1.0,
    hard_k: int = 256,
    rank_mode: str = "hinge",
    # early stop on AP
    patience: int = 25,
    min_delta: float = 1e-4,
):
    neg_mem = NegLogitMemory(capacity=mem_capacity, device=mem_device)

    best_ap = -1.0
    best_state = None
    best_info = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch_with_memory(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            asl_criterion=asl_criterion,
            neg_mem=neg_mem,
            device=device,
            accum_steps=accum_steps,
            clip_grad_norm=clip_grad_norm,
            rank_lambda=rank_lambda,
            rank_margin=rank_margin,
            hard_k=hard_k,
            rank_mode=rank_mode,
        )

        info = eval_ap_and_bestf1(model, val_loader, device)

        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_AP={info['ap']:.4f} val_bestF1={info['best_f1']:.4f} thr={info['best_thr']:.4f} "
            f"pos(mean/max)={info['pos_mean']:.4g}/{info['pos_max']:.4g} "
            f"neg(mean/max)={info['neg_mean']:.4g}/{info['neg_max']:.4g} "
            f"(pos/neg={info['pos_cnt']}/{info['neg_cnt']})"
        )

        # Early stop / best weight on AP (ranking metric)
        if info["ap"] > best_ap + min_delta:
            best_ap = info["ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_info = dict(info)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop. best_AP={best_ap:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_ap, best_info
    
    