"""Utility functions: seeding, evaluation, graph helpers."""
from __future__ import annotations

import os
import random
from typing import Tuple

import dgl
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return logits.argmax(dim=1).eq(labels).float().mean().item()


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def get_dense_adj(g: dgl.DGLGraph) -> torch.Tensor:
    """Return the dense adjacency matrix (without self-loops) of ``g``.

    Used by relational KD and OT subgraph helpers. Safe for the minibatch
    sizes used in OT/KD (<= ot_batch_size); avoid on full ogbn-arxiv.
    """
    g_nosl = dgl.remove_self_loop(g)
    src, dst = g_nosl.edges()
    n = g.num_nodes()
    A = torch.zeros(n, n, device=src.device)
    A[src, dst] = 1.0
    return A


def get_edge_endpoints(g: dgl.DGLGraph) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (src, dst) tensors of edges, with self-loops removed.

    Used by the edge-Dirichlet smoothness loss L_DE (eq 11).
    """
    g_nosl = dgl.remove_self_loop(g)
    return g_nosl.edges()


def sample_subgraph(
    g: dgl.DGLGraph,
    feats: torch.Tensor,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` nodes and induce a subgraph.

    Returns the subgraph, its node features, and the original node indices.
    """
    n = g.num_nodes()
    k = min(batch_size, n)
    idx = rng.choice(n, size=k, replace=False)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=feats.device)
    sub = g.subgraph(idx_t)
    return sub, feats[idx_t], idx_t


# ---------------------------------------------------------------------------
# Split helper (inductive setting; adapted from KRD)
# ---------------------------------------------------------------------------


def graph_split_inductive(
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    num_nodes: int,
    split_rate: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inductive split: a fraction of test nodes is hidden from the
    observed graph (idx_obs) and used as ``idx_test_ind``.
    """
    rng = np.random.default_rng(seed)
    test_perm = rng.permutation(idx_test.numpy())
    cut = int(len(test_perm) * split_rate)
    idx_test_ind = torch.as_tensor(test_perm[:cut], dtype=torch.long)
    idx_test_tran = torch.as_tensor(test_perm[cut:], dtype=torch.long)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    n1, n2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:n1]
    obs_idx_val = obs_idx_all[n1 : n1 + n2]
    obs_idx_test = obs_idx_all[n1 + n2 :]
    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind
