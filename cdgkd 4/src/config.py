"""Argparse configuration shared by pretrain.py and train.py."""
from __future__ import annotations

import argparse


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gnn", type=str, default="GCN", choices=["GCN", "SAGE", "GAT"])
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)


def get_pretrain_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GraphMAE pretraining on the source graph")
    add_common_args(p)
    p.add_argument("--source", type=str, required=True,
                   help="Source dataset name (e.g. cora, citeseer, pubmed, "
                        "amazon-photo, coauthor-cs, coauthor-phy, ogbn-arxiv).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-5)

    # GraphMAE-specific
    p.add_argument("--mask_rate", type=float, default=0.5,
                   help="Fraction of nodes whose features are corrupted "
                        "(0.5-0.75 typical).")
    p.add_argument("--replace_rate", type=float, default=0.1,
                   help="Of the masked nodes, this fraction get random "
                        "other nodes' features (noise injection). The rest "
                        "get the learnable [MASK] token.")
    p.add_argument("--alpha_l", type=float, default=2.0,
                   help="Scaled Cosine Error sharpness; higher focuses "
                        "training on harder reconstruction targets.")
    p.add_argument("--drop_edge_rate", type=float, default=0.0,
                   help="Optional edge-dropping rate applied to the graph "
                        "passed through the encoder (not the decoder).")

    p.add_argument("--batch_size", type=int, default=-1,
                   help="Random-subset minibatch size for large graphs. "
                        "<= 0 or >= num_nodes means full-graph training.")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--save_path", type=str, required=True,
                   help="Path to save the pretrained encoder state_dict.")
    p.add_argument("--data_dir", type=str, default="../data")
    return p.parse_args()


def get_train_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GOLD: cross-domain GNN-to-MLP distillation")
    add_common_args(p)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True,
                   help="Path to the SSL-pretrained source encoder.")
    p.add_argument("--exp_setting", type=str, default="tran",
                   choices=["tran", "ind"],
                   help="Transductive or inductive evaluation on target.")

    # Student MLP (f_theta + c_eta)
    p.add_argument("--mlp_layers", type=int, default=2)
    p.add_argument("--mlp_dropout", type=float, default=0.5)
    p.add_argument("--mlp_norm", type=str, default="none",
                   choices=["none", "batch", "layer"])

    # Adapter phi
    p.add_argument("--adapter_hidden", type=int, default=256)
    p.add_argument("--adapter_dropout", type=float, default=0.0)

    # Loss weights (eq 17-18; we decouple alpha_DE from lambda_align for
    # cleaner ablation - equivalent under lambda_DE := lambda_align * alpha_DE).
    p.add_argument("--lambda_ot", type=float, default=0.5,
                   help="Weight for L_OT (Sinkhorn divergence, eq 8).")
    p.add_argument("--lambda_de", type=float, default=0.1,
                   help="Weight for L_DE (edge Dirichlet energy, eq 11).")
    p.add_argument("--lambda_kd", type=float, default=1.0,
                   help="Weight for L_KD (relational neighborhood KL, eq 15).")

    # OT-specific (eq 8-10)
    p.add_argument("--ot_eps", type=float, default=0.05,
                   help="Sinkhorn entropic regularizer.")
    p.add_argument("--ot_sinkhorn_iter", type=int, default=50)
    p.add_argument("--ot_batch_size", type=int, default=256,
                   help="Number of nodes sampled per side per OT step.")

    # KD-specific (eq 13-15)
    p.add_argument("--kd_num_anchors", type=int, default=512,
                   help="Anchor nodes sampled per epoch for relational KD.")
    p.add_argument("--kd_num_pos", type=int, default=5,
                   help="Positives (neighbors) per anchor in C_i.")
    p.add_argument("--kd_num_neg", type=int, default=15,
                   help="Negatives (non-neighbors) per anchor in C_i.")
    p.add_argument("--kd_tau", type=float, default=0.5,
                   help="Temperature in the softmax over C_i (eq 13-14).")

    # Optimization
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--data_dir", type=str, default="../data")
    return p.parse_args()
