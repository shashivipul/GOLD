"""Model definitions for cross-domain GNN-to-MLP distillation.

Components:
    - GNN encoders (GCN / GAT / GraphSAGE) producing node embeddings.
      Adapted from KRD's models.py but stripped of the classifier head:
      output_dim = embedding dimension, not class count.
    - StudentMLP: MLP body + lightweight classifier head for the target task.
    - Adapter: target-feature -> source-feature space projector.
      Default is a 2-layer MLP; can be swapped for a GNN adapter.
    - ProjectionHead: 2-layer MLP used by SimGRACE during pretraining.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, GATConv


# ---------------------------------------------------------------------------
# GNN encoders (no classifier head; output is the node embedding)
# ---------------------------------------------------------------------------


class GCNEncoder(nn.Module):
    """Stacked GCN producing node embeddings of dimension ``output_dim``.

    ``output_dim`` defaults to ``hidden_dim``. Setting it lets the same
    class also serve as the GraphMAE decoder (hidden_dim -> input_dim).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        out_dim = output_dim if output_dim is not None else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, out_dim, activation=None))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=F.relu))
            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=F.relu))
            self.layers.append(GraphConv(hidden_dim, out_dim, activation=None))

    def forward(self, g, feats: torch.Tensor) -> torch.Tensor:
        h = feats
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = self.dropout(h)
        return h


class SAGEEncoder(nn.Module):
    """GraphSAGE encoder (gcn aggregator) producing node embeddings.

    Supports an explicit ``output_dim`` for use as a decoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        out_dim = output_dim if output_dim is not None else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, out_dim, aggregator_type="gcn"))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type="gcn"))
            for _ in range(num_layers - 2):
                self.layers.append(
                    SAGEConv(hidden_dim, hidden_dim, aggregator_type="gcn")
                )
            self.layers.append(SAGEConv(hidden_dim, out_dim, aggregator_type="gcn"))

    def forward(self, g, feats: torch.Tensor) -> torch.Tensor:
        h = feats
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GATEncoder(nn.Module):
    """GAT encoder with mean-pooling over heads at the last layer.

    Supports an explicit ``output_dim`` for use as a decoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_heads: int = 4,
        attn_drop: float = 0.3,
        negative_slope: float = 0.2,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        out_dim = output_dim if output_dim is not None else hidden_dim
        per_head = hidden_dim // num_heads
        self.layers = nn.ModuleList()
        heads = [num_heads] * num_layers + [1]
        self.layers.append(
            GATConv(
                input_dim,
                per_head,
                heads[0],
                dropout,
                attn_drop,
                negative_slope,
                residual=False,
                activation=F.relu,
            )
        )
        for l in range(1, num_layers - 1):
            self.layers.append(
                GATConv(
                    per_head * heads[l - 1],
                    per_head,
                    heads[l],
                    dropout,
                    attn_drop,
                    negative_slope,
                    residual=False,
                    activation=F.relu,
                )
            )
        # Final layer projects to out_dim with a single head.
        self.layers.append(
            GATConv(
                per_head * heads[-2] if num_layers > 1 else input_dim,
                out_dim,
                heads[-1],
                dropout,
                attn_drop,
                negative_slope,
                residual=False,
                activation=None,
            )
        )

    def forward(self, g, feats: torch.Tensor) -> torch.Tensor:
        h = feats
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
        return h


def build_encoder(
    name: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    output_dim: Optional[int] = None,
) -> nn.Module:
    """Construct an encoder by name. Pass ``output_dim`` to build a decoder.

    GraphMAE uses this factory twice: once with default ``output_dim``
    (= hidden_dim) for the encoder, and once with ``output_dim=input_dim``
    for the decoder.
    """
    name = name.upper()
    if name == "GCN":
        return GCNEncoder(input_dim, hidden_dim, num_layers, dropout, output_dim=output_dim)
    if name == "SAGE":
        return SAGEEncoder(input_dim, hidden_dim, num_layers, dropout, output_dim=output_dim)
    if name == "GAT":
        return GATEncoder(input_dim, hidden_dim, num_layers, dropout, output_dim=output_dim)
    raise ValueError(f"Unknown encoder: {name}")


# ---------------------------------------------------------------------------
# Student MLP: body produces embeddings, head produces logits
# ---------------------------------------------------------------------------


class StudentMLP(nn.Module):
    """MLP body (returns embeddings) plus a classifier head (returns logits).

    Use ``encode(x)`` to get embeddings (for KD alignment with the teacher),
    and ``forward(x)`` to get logits (for CE on target labels).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            # Degenerate case: a single linear that doubles as encoder + head.
            self.layers.append(nn.Linear(input_dim, hidden_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.norms.append(self._make_norm(norm_type, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.norms.append(self._make_norm(norm_type, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _make_norm(norm_type: str, dim: int) -> nn.Module:
        if norm_type == "batch":
            return nn.BatchNorm1d(dim)
        if norm_type == "layer":
            return nn.LayerNorm(dim)
        return nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate embedding (used for KD alignment)."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != self.num_layers - 1:
                if i < len(self.norms):
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits over target classes."""
        return self.classifier(self.encode(x))


# ---------------------------------------------------------------------------
# OT-based adapter and SimGRACE projection head
# ---------------------------------------------------------------------------


class Adapter(nn.Module):
    """Projects target features into the source feature space.

    A simple 2-layer MLP. The optimization signal (OT, KD, smoother) comes
    from outside; this module just provides the parametrization.
    """

    def __init__(
        self,
        target_dim: int,
        source_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(target_dim, source_dim)
        self.net = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, source_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
