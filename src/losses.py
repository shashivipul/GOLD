"""Loss functions for GOLD: GNN-to-MLP distillation from OOD teachers.

    - sce_loss:               GraphMAE scaled cosine error (Stage 1 pretraining).
    - sinkhorn_divergence:    debiased entropic OT  ->  L_OT  (eq 8, eq 10).
    - edge_dirichlet_energy:  per-edge smoothness   ->  L_DE  (eq 11).
    - relational_kd_loss:     neighborhood-distribution KL -> L_KD (eq 13-15).

All functions take batched / sub-graph inputs where appropriate.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage 1: GraphMAE reconstruction loss
# ---------------------------------------------------------------------------


def sce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    """Scaled Cosine Error (Hou et al., 2022).

        SCE(p, t) = mean_i (1 - cos(p_i, t_i))^alpha
    """
    p = F.normalize(pred, p=2, dim=-1)
    t = F.normalize(target, p=2, dim=-1)
    return (1.0 - (p * t).sum(dim=-1)).pow(alpha).mean()


# ---------------------------------------------------------------------------
# Stage 2: feature alignment (Sinkhorn divergence + Dirichlet energy)
# ---------------------------------------------------------------------------


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix between rows of x (m,d) and y (n,d)."""
    x_sq = (x * x).sum(dim=1, keepdim=True)
    y_sq = (y * y).sum(dim=1, keepdim=True).t()
    return (x_sq + y_sq - 2.0 * x @ y.t()).clamp(min=0.0)


def _sinkhorn_log(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    n_iter: int,
) -> torch.Tensor:
    """Log-domain Sinkhorn returning the transport plan T (m, n)."""
    log_a = torch.log(a + 1e-30).unsqueeze(1)
    log_b = torch.log(b + 1e-30).unsqueeze(0)
    log_K = -cost / eps
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v, dim=1, keepdim=True)
        log_v = log_b - torch.logsumexp(log_K + log_u, dim=0, keepdim=True)
    return torch.exp(log_K + log_u + log_v)


def _sinkhorn_ot_cost(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    n_iter: int,
) -> torch.Tensor:
    """Entropic OT cost <T*, M>. Differentiable in cost (no detached plan)."""
    T = _sinkhorn_log(cost, a, b, eps, n_iter)
    return (T * cost).sum()


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.05,
    n_iter: int = 50,
) -> torch.Tensor:
    """Debiased Sinkhorn divergence (eq 10 of the GOLD paper).

        S_eps(x, y) = OT_eps(x, y) - 0.5 * OT_eps(x, x) - 0.5 * OT_eps(y, y)

    Aligns the empirical source feature distribution mu_S with the adapted
    target feature distribution mu_{Z_T} as defined in eq 7-8 of the paper.
    """
    n_x, n_y = x.size(0), y.size(0)
    a = torch.full((n_x,), 1.0 / n_x, device=x.device, dtype=x.dtype)
    b = torch.full((n_y,), 1.0 / n_y, device=x.device, dtype=x.dtype)

    m_xy = _pairwise_sq_dist(x, y)
    m_xx = _pairwise_sq_dist(x, x)
    m_yy = _pairwise_sq_dist(y, y)

    ot_xy = _sinkhorn_ot_cost(m_xy, a, b, eps, n_iter)
    ot_xx = _sinkhorn_ot_cost(m_xx, a, a, eps, n_iter)
    ot_yy = _sinkhorn_ot_cost(m_yy, b, b, eps, n_iter)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy


def edge_dirichlet_energy(
    z: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
) -> torch.Tensor:
    """L_DE = (1 / |E_T|) * sum_{(i,j) in E_T} || z_i - z_j ||^2 (eq 11).

    Graph smoothness on adapter outputs. Pass each undirected edge once
    (the dataloader's edge_index already does this in DGL bidirectional
    form; we normalize by the count we get either way).
    """
    diff = z[edge_src] - z[edge_dst]
    return diff.pow(2).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Stage 2: topology-preserving relational KD (eq 13-15)
# ---------------------------------------------------------------------------


def _sample_candidate_sets(
    adj_dense: torch.Tensor,
    anchor_idx: torch.Tensor,
    num_pos: int,
    num_neg: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Build a (B, num_pos + num_neg) candidate-index tensor.

    For each anchor i:
        - num_pos positives sampled from N(i)   (with replacement if |N(i)| < num_pos)
        - num_neg negatives sampled uniformly from V \\ (N(i) U {i})

    Isolated anchors (|N(i)| = 0): positives are sampled from V \\ {i}, so
    they degrade gracefully to a near-uniform softmax target.
    """
    device = adj_dense.device
    n = adj_dense.size(0)
    B = anchor_idx.size(0)
    K = num_pos + num_neg

    cand = torch.empty(B, K, dtype=torch.long, device=device)
    # Build a self-mask once.
    self_idx = torch.arange(n, device=device)

    for b in range(B):
        i = anchor_idx[b].item()
        row = adj_dense[i]
        # Exclude self from neighbor mask.
        is_neighbor = row > 0
        is_neighbor[i] = False
        neighbors = is_neighbor.nonzero(as_tuple=False).squeeze(-1)
        non_neigh_mask = ~is_neighbor
        non_neigh_mask[i] = False
        non_neighbors = non_neigh_mask.nonzero(as_tuple=False).squeeze(-1)

        # ---- positives ----
        if neighbors.numel() == 0:
            # Degenerate: pull from anywhere except self.
            pool = self_idx[self_idx != i]
            idx_p = torch.randint(pool.numel(), (num_pos,), generator=rng, device=device)
            pos = pool[idx_p]
        elif neighbors.numel() >= num_pos:
            perm = torch.randperm(neighbors.numel(), generator=rng, device=device)
            pos = neighbors[perm[:num_pos]]
        else:
            idx_p = torch.randint(neighbors.numel(), (num_pos,), generator=rng, device=device)
            pos = neighbors[idx_p]

        # ---- negatives ----
        if non_neighbors.numel() >= num_neg:
            perm = torch.randperm(non_neighbors.numel(), generator=rng, device=device)
            neg = non_neighbors[perm[:num_neg]]
        elif non_neighbors.numel() > 0:
            idx_n = torch.randint(non_neighbors.numel(), (num_neg,), generator=rng, device=device)
            neg = non_neighbors[idx_n]
        else:
            # No non-neighbors at all (complete graph on the sub).
            pool = self_idx[self_idx != i]
            idx_n = torch.randint(pool.numel(), (num_neg,), generator=rng, device=device)
            neg = pool[idx_n]

        cand[b] = torch.cat([pos, neg])
    return cand


def relational_kd_loss(
    teacher_emb: torch.Tensor,
    student_emb: torch.Tensor,
    adj_dense: torch.Tensor,
    anchor_idx: torch.Tensor,
    num_pos: int,
    num_neg: int,
    tau: float,
    rng: torch.Generator,
) -> torch.Tensor:
    """Topology-preserving KD (eq 13-15).

        p_i^G(j) = softmax_{j in C_i}( cos(h^G_i, h^G_j) / tau )
        p_i^M(j) = softmax_{j in C_i}( cos(h^M_i, h^M_j) / tau )
        L_KD     = (1 / |anchors|) * sum_i KL( p_i^G || p_i^M )

    The candidate set C_i has fixed size num_pos + num_neg, built as
    described in _sample_candidate_sets. Teacher and student embeddings
    must live on the same target nodes (i.e. both indexed by V_T).

    Notes
    -----
    teacher_emb has gradient w.r.t. the adapter (since the teacher is
    fed adapter(X_T)); student_emb has gradient w.r.t. the student MLP.
    The KL is directed: forward KL with teacher as the reference.
    """
    cand = _sample_candidate_sets(adj_dense, anchor_idx, num_pos, num_neg, rng)

    # Normalize once.
    t_norm = F.normalize(teacher_emb, dim=1)
    s_norm = F.normalize(student_emb, dim=1)

    B, K = cand.shape
    anchor_t = t_norm[anchor_idx]                       # (B, d)
    anchor_s = s_norm[anchor_idx]                       # (B, d)
    cand_t = t_norm[cand.view(-1)].view(B, K, -1)       # (B, K, d)
    cand_s = s_norm[cand.view(-1)].view(B, K, -1)       # (B, K, d)

    sim_t = (anchor_t.unsqueeze(1) * cand_t).sum(dim=-1) / tau   # (B, K)
    sim_s = (anchor_s.unsqueeze(1) * cand_s).sum(dim=-1) / tau

    log_p_t = F.log_softmax(sim_t, dim=-1)
    log_p_s = F.log_softmax(sim_s, dim=-1)
    p_t = log_p_t.exp()

    # KL(p_t || p_s) per anchor, then mean.
    return (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()
