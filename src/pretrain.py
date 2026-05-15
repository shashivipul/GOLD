"""Stage 1: GraphMAE self-supervised pretraining on the source graph.

GraphMAE (Hou et al., 2022) trains a GNN encoder by masking a fraction of
node features and reconstructing them with a GNN decoder. The objective
explicitly forces the encoder to use the graph structure to predict missing
information, which makes the learned weights encode transferable structural
processing - exactly what Stage 2 needs.

Pipeline per step:

    1. Pick a random subset of nodes to mask (mask_rate).
    2. Of those: most get the learnable [MASK] token added to zeroed features;
       a small fraction (replace_rate) get a random other node's features
       (noise injection, slows the encoder from over-fitting to [MASK]).
    3. Optionally drop a fraction of edges (drop_edge_rate).
    4. Encode the corrupted graph -> per-node hidden representations.
    5. Pass hidden through ``encoder_to_decoder`` linear; zero-out the masked
       positions ("re-mask" trick - the decoder must infer them from neighbors).
    6. Decode -> reconstructed features.
    7. Scaled Cosine Error loss between reconstruction and original features
       at the masked positions only.

Only the encoder ``state_dict`` is saved. The decoder, ``encoder_to_decoder``
linear, and ``[MASK]`` token are discarded after Stage 1.
"""
from __future__ import annotations

import os
from typing import Tuple

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import get_pretrain_args
from dataloader import load_data
from losses import sce_loss
from models import build_encoder
from utils import ensure_dir, set_seed


class GraphMAEModule(nn.Module):
    """Encoder + encoder_to_decoder linear + decoder + [MASK] token."""

    def __init__(
        self,
        gnn_name: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            gnn_name, input_dim, hidden_dim, num_layers, dropout
        )
        # Encoder produces hidden_dim; decoder maps hidden_dim -> input_dim.
        self.decoder = build_encoder(
            gnn_name, hidden_dim, hidden_dim, num_layers, dropout,
            output_dim=input_dim,
        )
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Learnable [MASK] token added to corrupted node features.
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))


def encoding_mask_noise(
    feats: torch.Tensor,
    mask_token: torch.Tensor,
    mask_rate: float,
    replace_rate: float,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Corrupt features per GraphMAE.

    Most masked positions are replaced with the learnable [MASK] token;
    a small ``replace_rate`` fraction receive a random other node's features
    instead, which prevents the encoder from over-fitting to the [MASK] token.

    Returns:
        corrupted_feats: (N, d) corrupted feature matrix.
        mask_nodes:      indices of nodes whose features were masked.
    """
    n = feats.size(0)
    num_mask = int(mask_rate * n)
    if num_mask == 0:
        return feats.clone(), torch.empty(0, dtype=torch.long, device=feats.device)

    perm = torch.randperm(n, generator=rng, device=feats.device)
    mask_nodes = perm[:num_mask]

    out = feats.clone()
    if replace_rate > 0:
        token_count = int((1.0 - replace_rate) * num_mask)
        perm_mask = torch.randperm(num_mask, generator=rng, device=feats.device)
        token_nodes = mask_nodes[perm_mask[:token_count]]
        noise_nodes = mask_nodes[perm_mask[token_count:]]
        noise_count = noise_nodes.numel()
        noise_src = torch.randperm(n, generator=rng, device=feats.device)[:noise_count]
        out[token_nodes] = 0.0
        out[noise_nodes] = feats[noise_src]
        out[token_nodes] = out[token_nodes] + mask_token
    else:
        out[mask_nodes] = 0.0
        out[mask_nodes] = out[mask_nodes] + mask_token

    return out, mask_nodes


def drop_edges(g: dgl.DGLGraph, drop_rate: float, gen: torch.Generator) -> dgl.DGLGraph:
    """Randomly drop edges; returns a new graph on the same device."""
    if drop_rate <= 0:
        return g
    num_edges = g.num_edges()
    keep_mask = torch.rand(num_edges, generator=gen, device=g.device) >= drop_rate
    edges_to_drop = (~keep_mask).nonzero(as_tuple=False).squeeze(1)
    if edges_to_drop.numel() == 0:
        return g
    return dgl.remove_edges(g, edges_to_drop)


def main() -> None:
    args = get_pretrain_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(args.seed)

    # ----- data -----
    g, _, _, _, _ = load_data(args.source, seed=args.seed)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g).to(device)
    feats = g.ndata["feat"].to(device) if "feat" in g.ndata else g.ndata["features"].to(device)
    input_dim = feats.shape[1]

    # ----- model -----
    model = GraphMAEModule(
        gnn_name=args.gnn,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[pretrain] GraphMAE | dataset={args.source} gnn={args.gnn} "
          f"input_dim={input_dim} hidden_dim={args.hidden_dim} "
          f"nodes={g.num_nodes()} edges={g.num_edges()} "
          f"mask_rate={args.mask_rate} replace_rate={args.replace_rate}")

    best_loss = float("inf")
    waited = 0
    rng_np = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Optional minibatch via random subset (for ogbn-arxiv-scale graphs).
        if args.batch_size > 0 and args.batch_size < g.num_nodes():
            idx = torch.as_tensor(
                rng_np.choice(g.num_nodes(), size=args.batch_size, replace=False),
                dtype=torch.long, device=device,
            )
            cur_g = g.subgraph(idx)
            cur_feats = feats[idx]
        else:
            cur_g = g
            cur_feats = feats

        # Optional edge dropping on the working graph (for the encoder pass only).
        cur_g_for_enc = drop_edges(cur_g, args.drop_edge_rate, gen)

        # 1-2. Corrupt features.
        corrupted, mask_nodes = encoding_mask_noise(
            cur_feats, model.enc_mask_token,
            mask_rate=args.mask_rate,
            replace_rate=args.replace_rate,
            rng=gen,
        )

        # 3-4. Encode corrupted graph.
        h = model.encoder(cur_g_for_enc, corrupted)

        # 5. encoder -> decoder bridge + re-mask the masked positions to zero,
        # forcing the decoder to infer them from neighbors.
        h_dec = model.encoder_to_decoder(h)
        if mask_nodes.numel() > 0:
            h_dec = h_dec.clone()
            h_dec[mask_nodes] = 0.0

        # 6. Decode on the original (un-edge-dropped) graph.
        recon = model.decoder(cur_g, h_dec)

        # 7. SCE loss only on the masked nodes.
        if mask_nodes.numel() == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = sce_loss(recon[mask_nodes], cur_feats[mask_nodes], alpha=args.alpha_l)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"[pretrain] ep {epoch:4d}  sce={loss.item():.6f}  "
              f"masked={mask_nodes.numel()}/{cur_feats.size(0)}")

        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            waited = 0
            ensure_dir(args.save_path)
            # Save ONLY the encoder weights. Decoder, encoder_to_decoder linear,
            # and the [MASK] token are scaffolding and stay in Stage 1.
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "args": vars(args),
                    "input_dim": input_dim,
                    "pretrain_method": "GraphMAE",
                },
                args.save_path,
            )
        else:
            waited += 1
            if waited >= args.patience:
                print(f"[pretrain] early stopping at epoch {epoch}")
                break

    print(f"[pretrain] best loss={best_loss:.6f}; saved to {args.save_path}")


if __name__ == "__main__":
    main()
