# GOLD: GNN-to-MLP KD from Out-of-Distribution Teachers

Reference implementation of GOLD (paper anonymous; under review at TMLR).

A GNN teacher is pretrained on a **source graph** with self-supervised GraphMAE,
frozen, and used to supervise an MLP student deployed on a **target graph** with
graph-free inference. A learned target-to-source feature adapter, regularized by
optimal-transport alignment (Sinkhorn divergence) and edge-Dirichlet smoothness,
makes the frozen teacher's representations meaningful on target data. A
topology-preserving relational KD objective transfers neighborhood structure
from the teacher into the student.

## Pipeline

```
Stage 1   ── G^S, X^S ─────► GraphMAE pretraining ─────► frozen encoder g^S
                                                                  │
Stage 2                                                           │
                                                                  ▼
        G^T, X^T ─► phi(X^T) = Z^T ─┬─► g^S(A^T, Z^T) = H^G  (teacher; frozen)
                                    │
                                    └─► f_theta(Z^T)  = H^M  (student; trained)
                                                          │
                                                          └─► c_eta(H^M) = Y_hat

Deploy  (graph-free):
        x^T_i ─► phi ─► f_theta ─► c_eta ─► y_hat_i
```

## Losses (Stage 2)

```
L_GOLD =  L_task                                              (CE on target labels)
        + lambda_ot  · L_OT     (Sinkhorn divergence on  mu_S, mu_{Z_T})
        + lambda_de  · L_DE     (edge Dirichlet energy on  Z_T)
        + lambda_kd  · L_KD     (relational neighborhood KL via candidate sets)
```

Equation numbers in `src/train.py` and `src/losses.py` reference the paper
directly. The paper writes the smoothness coupled to alignment as
`lambda_align (L_OT + alpha_DE L_DE)`; the code decouples them for cleaner
ablations. Equivalent under `lambda_de := lambda_align * alpha_DE`.

## Repository layout

```
cdgkd/
├── src/
│   ├── models.py            # GNN encoders, MLP student, adapter phi
│   ├── losses.py            # sce, sinkhorn_divergence, edge_dirichlet,
│   │                          relational_kd
│   ├── pretrain.py          # Stage 1: GraphMAE on the source graph
│   ├── train.py             # Stage 2: GOLD (Algorithm 1)
│   ├── dataloader.py        # KRD's loaders (DGL)
│   ├── data_preprocess.py   # KRD's CPF preprocessor (unchanged)
│   ├── utils.py             # seeding, eval, dense-adj, edge helpers
│   └── config.py            # argparse defaults aligned with paper notation
├── configs/default.yaml
├── scripts/run_example.sh
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place KRD's `data/` folder (the CPF `.npz` files) at the repo root or symlink it.

## Usage

```bash
bash scripts/run_example.sh
```

Or run the stages manually -- see `scripts/run_example.sh` for the full
argument set; defaults match `configs/default.yaml`.


