"""Stage 2: GOLD cross-domain knowledge distillation.

Implements Algorithm 1 of the GOLD paper:

    L_GOLD = L_task + lambda_OT * L_OT + lambda_DE * L_DE + lambda_KD * L_KD

(The paper's eq 18 nests L_DE inside lambda_align; we decouple it for cleaner
ablations - see paper sense-check note 9. Equivalent under reparameterization
lambda_DE = lambda_align * alpha_DE.)

Forward per step:
    Z_T   = phi(X_T)                       (adapter; eq 3)
    H_T^G = g^S(A_T, Z_T)    (frozen)      (teacher embeddings on target; eq 4)
    H_T^M = f_theta(Z_T)                   (student embeddings; eq 5)
    Y_hat = c_eta(H_T^M)                   (target logits; eq 6)

    L_task = CE(Y_hat[V_T^L], Y_T^L)             (eq 16)
    L_OT   = S_eps( mu_S, mu_{Z_T} )             (eq 8)
    L_DE   = mean_{(i,j) in E_T} ||z_i - z_j||^2 (eq 11)
    L_KD   = mean_i KL( p_i^G || p_i^M )         (eq 15)

Deployed predictor: y_hat_i = c_eta(f_theta(phi(x_T_i)))  --  graph-free (eq 19).
"""
from __future__ import annotations

from typing import Tuple

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from config import get_train_args
from dataloader import load_data
from losses import (
    edge_dirichlet_energy,
    relational_kd_loss,
    sinkhorn_divergence,
)
from models import Adapter, StudentMLP, build_encoder
from utils import (
    accuracy,
    get_dense_adj,
    get_edge_endpoints,
    sample_subgraph,
    set_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_feats(g) -> torch.Tensor:
    if "feat" in g.ndata:
        return g.ndata["feat"]
    if "features" in g.ndata:
        return g.ndata["features"]
    raise KeyError("No node features found on graph.")


def load_teacher(
    ckpt_path: str,
    gnn_name: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    """Load the SSL-pretrained source encoder. Returns (encoder, source_input_dim).

    The encoder's parameters are frozen and the module is set to eval() so
    dropout / batchnorm behave deterministically. Gradient still flows from
    inputs (adapted target features) back through the teacher to the adapter.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    source_input_dim = ckpt["input_dim"]
    encoder = build_encoder(
        gnn_name, source_input_dim, hidden_dim, num_layers, dropout
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"[train] loaded {ckpt.get('pretrain_method', 'SSL')} teacher; "
          f"source input_dim={source_input_dim}")
    return encoder, source_input_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = get_train_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng_torch = torch.Generator(device=device).manual_seed(args.seed)
    rng_np = np.random.default_rng(args.seed)

    # ----- target graph (for KD, smoothness, task supervision) -----
    g_t, y_t, idx_train, idx_val, idx_test = load_data(args.target, seed=args.seed)
    g_t = dgl.remove_self_loop(g_t)
    g_t = dgl.add_self_loop(g_t).to(device)
    x_t = _get_feats(g_t).to(device)
    y_t = y_t.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    num_classes = int(y_t.max().item() + 1)

    target_dim = x_t.shape[1]
    edge_src_t, edge_dst_t = get_edge_endpoints(g_t)

    # ----- source graph (for OT alignment) -----
    g_s, _, _, _, _ = load_data(args.source, seed=args.seed)
    g_s = dgl.remove_self_loop(g_s)
    g_s = dgl.add_self_loop(g_s).to(device)
    x_s = _get_feats(g_s).to(device)

    # ----- teacher: frozen GraphMAE encoder -----
    teacher, source_dim = load_teacher(
        args.teacher_ckpt, args.gnn, args.hidden_dim,
        args.num_layers, args.dropout, device,
    )

    # ----- adapter phi: target_dim -> source_dim -----
    adapter = Adapter(
        target_dim=target_dim,
        source_dim=source_dim,
        hidden_dim=args.adapter_hidden,
        dropout=args.adapter_dropout,
    ).to(device)

    # ----- student: input is Z_T (source_dim), output is hidden_dim -----
    # Per the GOLD paper, both teacher and student consume Z_T = phi(X_T).
    student = StudentMLP(
        input_dim=source_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.mlp_layers,
        dropout=args.mlp_dropout,
        norm_type=args.mlp_norm,
    ).to(device)

    trainable_params = list(adapter.parameters()) + list(student.parameters())
    optim = Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"[train] GOLD | source={args.source} ({x_s.shape[0]} nodes, {source_dim}d) "
        f" target={args.target} ({x_t.shape[0]} nodes, {target_dim}d) "
        f" num_classes={num_classes} hidden_dim={args.hidden_dim}"
    )

    # Cache the dense target adjacency for relational-KD anchor sampling.
    # For ogbn-arxiv-scale graphs this is unsafe; in that regime, switch to
    # an in-batch dense adj over sampled anchors (see paper sense-check note).
    dense_A_t = get_dense_adj(g_t)

    best_val_acc = 0.0
    best_test_acc = 0.0
    waited = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        adapter.train()

        # ---- Algorithm 1, lines 2-5 -------------------------------------
        z_t = adapter(x_t)                          # (n_T, d_S) -- Z_T = phi(X_T)
        teacher_emb = teacher(g_t, z_t)             # H_T^G  (eq 4); teacher frozen
        student_emb = student.encode(z_t)           # H_T^M  (eq 5)
        student_logits = student.classifier(student_emb)  # Y_hat (eq 6)

        # ---- Algorithm 1, line 6: L_task --------------------------------
        loss_task = F.cross_entropy(student_logits[idx_train], y_t[idx_train])

        # ---- Algorithm 1, line 7: L_OT ----------------------------------
        # Minibatch the OT alignment for tractability on large graphs.
        sub_s, sub_x_s, _ = sample_subgraph(g_s, x_s, args.ot_batch_size, rng_np)
        sub_t, sub_x_t_raw, _ = sample_subgraph(g_t, x_t, args.ot_batch_size, rng_np)
        sub_z_t = adapter(sub_x_t_raw)
        loss_ot = sinkhorn_divergence(
            sub_x_s, sub_z_t,
            eps=args.ot_eps,
            n_iter=args.ot_sinkhorn_iter,
        )

        # ---- Algorithm 1, line 8: L_DE (eq 11) --------------------------
        loss_de = edge_dirichlet_energy(z_t, edge_src_t, edge_dst_t)

        # ---- Algorithm 1, lines 9-10: relational KD (eq 13-15) ----------
        anchor_idx = torch.as_tensor(
            rng_np.choice(g_t.num_nodes(), size=args.kd_num_anchors, replace=False),
            dtype=torch.long, device=device,
        )
        loss_kd = relational_kd_loss(
            teacher_emb=teacher_emb,
            student_emb=student_emb,
            adj_dense=dense_A_t,
            anchor_idx=anchor_idx,
            num_pos=args.kd_num_pos,
            num_neg=args.kd_num_neg,
            tau=args.kd_tau,
            rng=rng_torch,
        )

        # ---- Algorithm 1, line 11: L_GOLD (eq 18, decoupled DE) ---------
        loss = (
            loss_task
            + args.lambda_ot * loss_ot
            + args.lambda_de * loss_de
            + args.lambda_kd * loss_kd
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ---- eval (graph-free predictor, eq 19) -------------------------
        student.eval()
        adapter.eval()
        with torch.no_grad():
            logits_eval = student(adapter(x_t))     # c_eta o f_theta o phi
            val_acc = accuracy(logits_eval[idx_val], y_t[idx_val])
            test_acc = accuracy(logits_eval[idx_test], y_t[idx_test])

        print(
            f"[ep {epoch:4d}] task={loss_task.item():.4f} "
            f"ot={loss_ot.item():.4f} de={loss_de.item():.4f} kd={loss_kd.item():.4f} "
            f"| val={val_acc:.4f} test={test_acc:.4f} "
            f"(best_val={best_val_acc:.4f}, test@best_val={best_test_acc:.4f})"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            waited = 0
        else:
            waited += 1
            if waited >= args.patience:
                print(f"[train] early stopping at epoch {epoch}")
                break

    print(f"[train] DONE.  best_val={best_val_acc:.4f}  "
          f"test@best_val={best_test_acc:.4f}")


if __name__ == "__main__":
    main()
