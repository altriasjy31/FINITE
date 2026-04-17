#!/usr/bin/env python3
"""
Train a simple fusion network for GO scoring.

Inputs:
  - Three score matrices on TRAIN split: shape (N, C) each
  - Three score matrices on VAL split:   shape (N_val, C) each
  - Multi-label ground-truth matrices:   shape (N, C) and (N_val, C)

Model:
  - Project each channel (model) from C -> rank -> hidden
  - Fuse 3 channels in hidden space via gating (softmax weights)
  - Decode hidden -> C logits (full GO output, enables full ranking)

Loss:
  - AsymmetricLoss or AsymmetricLossOptimized (from loss_functions.loss)

Optimizer & schedule:
  - Adam or AdamW
  - OneCycleLR

Notes:
  - This script trains on full-GO outputs (no candidate pruning).
  - For BP (C ~ 21000), use npy memmap to avoid loading everything into RAM.
"""

import os
import sys
import pathlib as P
prj_root = P.Path(__file__).parent.parent
if (t := str(prj_root)) not in sys.path:
    sys.path.insert(0, t)
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import numpy as np
from scipy.integrate import trapezoid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from timm.optim import create_optimizer_v2

from models.fuse_nn import (FusionProjectThenFuse, 
                            FusionTransformerLatent, 
                            FusionLinearBaseTransformerResidual,
                            NGHFusion)
from util.loss import AsymmetricLoss, AsymmetricLossOptimized
from util.moving_average import ModelEma

def mixup_batch(x, y, e=None, alpha=0.2):
    """
    Mixup for multi-label classification.

    Args:
      x: (B, 3, C) float tensor
      y: (B, C) float tensor (0/1 or soft)
      alpha: Beta distribution parameter

    Returns:
      x_mix, y_mix
    """
    if alpha <= 0:
        return x, y

    lam = np.random.beta(alpha, alpha)
    b = x.size(0)
    perm = torch.randperm(b, device=x.device)

    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_mix = lam * y + (1.0 - lam) * y[perm]
    if e is not None:
        e_mix = lam * e + (1.0 - lam) * e[perm]
        return x_mix, y_mix, e_mix
    return x_mix, y_mix

class FusionUnionSpaceDataset(Dataset):
    """
    Dataset for union-space fusion training.

    Directory structure (example):
      root/go_name/split/
        - target_preds_union.npz   (keys: zhao, esm2_v3, profunsom; each (N, C))
        - union_labels.npy         ((N, C) binary)
        - common_proteins.npy      ((N,) protein ids)
        - union_go_terms.npy       ((C,) go terms)

    Each item:
      x: (3, C) float32
      y: (C,)  float32 {0,1}
    """
    def __init__(self, split_dir: str, mmap_labels: bool = True):
        super().__init__()
        self.split_dir = split_dir

        preds_path = os.path.join(split_dir, "scores_union.npy")
        labels_path = os.path.join(split_dir, "union_labels.npy")
        proteins_path = os.path.join(split_dir, "common_proteins.npy")
        go_terms_path = os.path.join(split_dir, "union_go_terms.npy")

        if not os.path.isfile(preds_path):
            raise FileNotFoundError(f"Missing file: {preds_path}")
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"Missing file: {labels_path}")
        if not os.path.isfile(proteins_path):
            raise FileNotFoundError(f"Missing file: {proteins_path}")
        if not os.path.isfile(go_terms_path):
            raise FileNotFoundError(f"Missing file: {go_terms_path}")

        # --- Load predictions (npz) ---
        # Note: np.load(npz) does not support true memmap; it is loaded into memory.
        # preds = np.load(preds_path, allow_pickle=False)
        # required_keys = ("zhao", "esm2_v3", "profunsom")
        # for k in required_keys:
        #     if k not in preds:
        #         raise KeyError(f"Key '{k}' not found in {preds_path}. Available keys: {list(preds.keys())}")

        # self.s_zhao = preds["zhao"]
        # self.s_esm2 = preds["esm2_v3"]
        # self.s_som = preds["profunsom"]
        preds = np.load(preds_path, mmap_mode="r" if mmap_labels else None)
        self.s_zhao, self.s_esm2, self.s_som = preds[:, 0, :], preds[:, 1, :], preds[:, 2, :]

        if self.s_zhao.shape != self.s_esm2.shape or self.s_zhao.shape != self.s_som.shape:
            raise ValueError(
                f"Score matrices shape mismatch: "
                f"zhao={self.s_zhao.shape}, esm2_v3={self.s_esm2.shape}, profunsom={self.s_som.shape}"
            )

        # --- Load labels (npy) ---
        self.y = np.load(labels_path, mmap_mode="r" if mmap_labels else None)

        if self.y.shape != self.s_zhao.shape:
            raise ValueError(f"Label shape mismatch: labels={self.y.shape}, scores={self.s_zhao.shape}")

        # --- Load proteins / go terms (optional for training, useful for debugging) ---
        self.proteins = np.load(proteins_path, allow_pickle=True)
        self.go_terms = np.load(go_terms_path, allow_pickle=True)

        self.n, self.c = self.s_zhao.shape

        if len(self.proteins) != self.n:
            raise ValueError(f"Protein count mismatch: proteins={len(self.proteins)} vs N={self.n}")
        if len(self.go_terms) != self.c:
            raise ValueError(f"GO term count mismatch: go_terms={len(self.go_terms)} vs C={self.c}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        # (C,)
        s1 = self.s_zhao[idx]
        s2 = self.s_esm2[idx]
        s3 = self.s_som[idx]
        y = self.y[idx]

        # Stack to (3, C)
        x = np.stack([s1, s2, s3], axis=0).astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        return torch.from_numpy(x), torch.from_numpy(y)

class H5IndexDataset(Dataset):
    def __init__(self, h5_path: str,
                 prot_emb_path = None):
        super().__init__()
        self.h5_path = h5_path
        self._h5 = None  # lazy-open per worker

        with h5py.File(self.h5_path, "r") as f:
            if "scores" not in f or "labels" not in f:
                raise KeyError(f"H5 must contain 'scores' and 'labels'. Keys: {list(f.keys())}")
            n, ch, c = f["scores"].shape
            if ch != 3:
                raise ValueError(f"Expected scores shape (N,3,C), got {f['scores'].shape}")
            if f["labels"].shape != (n, c):
                raise ValueError(f"labels shape mismatch: {f['labels'].shape} vs {(n, c)}")
            self.n = int(n)
            self.c = int(c)

        self.prot_emb = None
        self.prot_emb_dim = None
        if prot_emb_path is not None:
            emb = np.load(prot_emb_path, mmap_mode="r")   # 避免重复拷贝整块内存
            if emb.shape[0] != self.n:
                raise ValueError(
                    f"prot_emb rows {emb.shape[0]} != H5 samples {self.n}"
                )
            self.prot_emb = emb
            self.prot_emb_dim = emb.shape[1]

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        return int(idx)

    def _get_h5(self):
        # Each worker process must open its own handle
        if self._h5 is None:
            # swmr=True helps concurrent reads
            self._h5 = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        return self._h5

    def collate_fn(self, batch_indices):
        """
        Batch read from HDF5 with sorted indices (required by h5py),
        then restore the original order to preserve shuffling semantics.
        """
        h5 = self._get_h5()

        idx = np.asarray(batch_indices, dtype=np.int64)

        # h5py fancy indexing requires indices to be strictly increasing
        order = np.argsort(idx)
        idx_sorted = idx[order]

        # Read in sorted order
        scores_sorted = h5["scores"][idx_sorted]  # (B,3,C)
        labels_sorted = h5["labels"][idx_sorted]  # (B,C)

        emb_sorted = None
        if self.prot_emb is not None:
            emb_sorted = self.prot_emb[idx_sorted]  # (B, Dp)

        # Restore original order
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))

        scores = scores_sorted[inv]
        labels = labels_sorted[inv]
        if emb_sorted is not None:
            embs = emb_sorted[inv]
        else:
            embs = None

        x = torch.tensor(scores, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        if embs is not None:
            e = torch.tensor(embs, dtype=torch.float32)
        else:
            e = None

        return x, y, e


    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

                
# -----------------------------
#  Metrics (simple reference)
# -----------------------------
@torch.no_grad()
def eval_fmax_aupr(y_true: np.ndarray, y_prob: np.ndarray, num_thresholds: int = 101) -> Tuple[float, float]:
    """
    Evaluate Fmax and AUPR in a straightforward way.
    - y_true: (N, C) {0,1}
    - y_prob: (N, C) [0,1]
    """
    eps = 1e-12
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    # AUPR (micro)
    # Sort by probability descending
    flat_true = y_true.reshape(-1).astype(np.int32)
    flat_prob = y_prob.reshape(-1)
    order = np.argsort(-flat_prob)
    flat_true = flat_true[order]
    tp = np.cumsum(flat_true)
    fp = np.cumsum(1 - flat_true)
    prec = tp / (tp + fp + eps)
    rec = tp / (flat_true.sum() + eps)
    # Trapezoidal integration over recall
    aupr = trapezoid(prec, rec)

    # Fmax (protein-centric micro-averaged F over all predictions)
    best_f = 0.0
    for t in thresholds:
        pred = (y_prob >= t).astype(np.int32)
        tp = (pred * y_true).sum()
        fp = (pred * (1 - y_true)).sum()
        fn = ((1 - pred) * y_true).sum()
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f = 2 * p * r / (p + r + eps)
        if f > best_f:
            best_f = f

    return float(best_f), float(aupr)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_true = []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        if len(batch) == 3:
            x, y, e = batch
            x = x.to(device, non_blocking=True)  # (B,3,C)
            y = y.to(device, non_blocking=True)  # (B,C)
            e = e.to(device, non_blocking=True)
        else:
            x, y = batch
            x = x.to(device, non_blocking=True)  # (B,3,C)
            y = y.to(device, non_blocking=True)  # (B,C)
            e = None

        logits, _ = model(x, e)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    fmax, aupr = eval_fmax_aupr(y_true, y_prob)
    return {"fmax": fmax, "aupr": aupr}


# -----------------------------
#  Training utilities
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loss(loss_name: str, **kwargs):
    if loss_name == "asl":
        return AsymmetricLoss(**kwargs)
    if loss_name == "asl_opt":
        return AsymmetricLossOptimized(**kwargs)
    raise ValueError(f"Unknown loss: {loss_name}")


def build_optimizer(model: nn.Module, opt_name: str, lr: float, weight_decay: float):
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "muon":
        return create_optimizer_v2(model, "muon", lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {opt_name}")

def kd_loss_logits_masked(student_logits, teacher_logits, T=2.0, p_low=0.05, p_high=0.7):
    s = student_logits / T
    t = teacher_logits / T
    p_s = torch.sigmoid(s)
    p_t = torch.sigmoid(t)

    # 只在 teacher 较不确定的 term 上做 KD
    mask = (p_t > p_low) & (p_t < p_high)  # (B,C) bool
    if mask.sum() == 0:
        # fallback: no mask
        return F.binary_cross_entropy_with_logits(p_s, p_t, reduction="mean") * (T * T)

    loss = F.binary_cross_entropy_with_logits(
        p_s[mask], p_t[mask], reduction="mean"
    )
    return loss * (T * T)

# -----------------------------
#  Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Train fusion network (project-then-fuse) on union-space dataset.")

    # Dataset layout
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of union-space dataset, e.g. ./results/union_space_preds_only",
    )
    p.add_argument(
        "--go_name",
        type=str,
        required=True,
        choices=["bp", "cc", "mf"],
        help="Ontology namespace.",
    )
    p.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Train split folder name under go_name (default: train).",
    )
    p.add_argument(
        "--valid_split",
        type=str,
        default="valid",
        help="Validation split folder name under go_name (default: valid).",
    )

    # Backend selection
    p.add_argument(
        "--backend",
        type=str,
        default="npz",
        choices=["npz", "npy", "h5"],
        help=(
            "Data backend: "
            "'npz' reads target_preds_union.npz + union_labels.npy; "
            "'npy' reads scores_union.npy + labels_union.npy; "
            "'h5' reads a single HDF5 file (provide --train_h5/--valid_h5)."
        ),
    )
    p.add_argument(
        "--mmap",
        action="store_true",
        help="Use memmap when backend=npy (recommended for BP). Ignored for npz.",
    )

    # For HDF5 backend
    p.add_argument("--train_h5", type=str, default=None, help="Train HDF5 path (backend=h5).")
    p.add_argument("--valid_h5", type=str, default=None, help="Valid HDF5 path (backend=h5).")

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)

    # Model hyperparams
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    """
    --latent_tokens 8

    --num_heads 8

    --depth 2

    --learn_base_weights（bool）

    --res_scale 0.5

    --use_logit_input（默认 True）
    """
    p.add_argument("--latent_tokens", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--res_scale", type=float, default=0.5)
    p.add_argument("--train_go_features", type=str,)
    p.add_argument("--valid_go_features", type=str,)
    p.add_argument("--train_prot_features", type=str)
    p.add_argument("--valid_prot_features", type=str)
    # NOTE: Default True to avoid train/infer mismatch.
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--use_logit_input",
        dest="use_logit_input",
        action="store_true",
        default=True,
        help="Treat input scores as probabilities in [0,1] and convert to logits.",
    )
    g.add_argument(
        "--no_logit_input",
        dest="use_logit_input",
        action="store_false",
        help="Treat input scores as logits already (skip probability->logit conversion).",
    )
    p.set_defaults(use_logit_input=True)  # choose a default; bash will still pass explicit flag

    p.add_argument("--use_term_gate", action="store_true", help="Enable per-term (GO-term) channel gating.")
    p.add_argument("--no_term_gate", dest="use_term_gate", action="store_false", help="Disable per-term gating.")
    p.set_defaults(use_term_gate=True)

    p.add_argument("--gate_scale_init", type=float, default=1.0, help="Initial scale for term gate logits.")
    p.add_argument("--gate_reg_lambda", type=float, default=0.05, help="L2 regularization strength to keep term gate close to global base weights.")


    # Augment
    p.add_argument("--mixup_p", type=float, default=0.3, help="Probability to apply mixup per batch.")
    p.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup Beta(alpha, alpha).")

    # Distillation
    p.add_argument(
        "--use_kd", action="store_true",
        help="Enable knowledge distillation from linear fusion teacher."
    )
    p.add_argument("--use_ema_kd", action="store_true")
    p.add_argument(
        "--kd_lambda", type=float, default=0.2,
        help="Weight for KD loss term."
    )
    p.add_argument("--kd_T", type=float, default=2.0)

    p.add_argument(
        "--teacher_weights", nargs=3, type=float, default=None,
        help="Base fusion weights (zhao, esm2_v3, profunsom)."
    )

    p.add_argument("--num_decode_queries", type=int)

    # Optimization
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "muon"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # OneCycleLR
    p.add_argument("--pct_start", type=float, default=0.1)
    p.add_argument("--div_factor", type=float, default=25.0)
    p.add_argument("--final_div_factor", type=float, default=1e4)

    # Loss (Asymmetric Loss)
    p.add_argument("--loss", type=str, default="asl_opt", choices=["asl", "asl_opt"])
    p.add_argument("--gamma_neg", type=float, default=4.0)
    p.add_argument("--gamma_pos", type=float, default=1.0)
    p.add_argument("--clip", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--disable_torch_grad_focal_loss", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./fusion_ckpt")
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--amp", action="store_true", help="Use mixed precision AMP.")
    p.add_argument("--val_max_batches", type=int, default=None, help="Limit val batches for quick debugging.")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Resolve split directories
    # -------------------------
    train_dir = os.path.join(args.data_root, args.go_name, args.train_split)
    valid_dir = os.path.join(args.data_root, args.go_name, args.valid_split)

    if args.backend in ("npz", "npy"):
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Train split dir not found: {train_dir}")
        if not os.path.isdir(valid_dir):
            raise FileNotFoundError(f"Valid split dir not found: {valid_dir}")

    # -------------------------
    # Build datasets/loaders
    # -------------------------
    if args.backend == "npz":
        # Uses:
        #   target_preds_union.npz (zhao/esm2_v3/profunsom) + union_labels.npy
        train_ds = FusionUnionSpaceDataset(train_dir, mmap_labels=True)
        val_ds = FusionUnionSpaceDataset(valid_dir, mmap_labels=True)

        # Ensure identical GO term order
        if not np.array_equal(train_ds.go_terms, val_ds.go_terms):
            raise ValueError("union_go_terms.npy differs between train and valid. Align them before training.")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        num_classes = train_ds.c

    elif args.backend == "npy":
        # Expected files in split dir:
        #   scores_union.npy (N,3,C) + labels_union.npy (N,C)
        # You need a corresponding dataset class. Example name below:
        #   FusionUnionSpaceNpyDataset(split_dir, mmap=args.mmap)
        train_ds = FusionUnionSpaceDataset(train_dir, mmap=args.mmap)
        val_ds = FusionUnionSpaceDataset(valid_dir, mmap=args.mmap)

        if not np.array_equal(train_ds.go_terms, val_ds.go_terms):
            raise ValueError("union_go_terms.npy differs between train and valid. Align them before training.")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        num_classes = train_ds.c

    else:  # args.backend == "h5"
        if args.train_h5 is None or args.valid_h5 is None:
            raise ValueError("For backend=h5, you must provide --train_h5 and --valid_h5.")
        if not os.path.isfile(args.train_h5):
            raise FileNotFoundError(f"Train HDF5 not found: {args.train_h5}")
        if not os.path.isfile(args.valid_h5):
            raise FileNotFoundError(f"Valid HDF5 not found: {args.valid_h5}")

        train_ds = H5IndexDataset(args.train_h5, args.train_prot_features)
        val_ds = H5IndexDataset(args.valid_h5, args.valid_prot_features)

        if train_ds.c != val_ds.c:
            raise ValueError(f"Train/Valid C mismatch: train C={train_ds.c}, valid C={val_ds.c}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=train_ds.collate_fn,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=val_ds.collate_fn,
            drop_last=False,
        )
        num_classes = train_ds.c

    # -------------------------
    # Model, loss, optimizer
    # -------------------------
    model = NGHFusion(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,          # e.g. 512
        latent_tokens=args.latent_tokens,    # e.g. 8
        num_heads=args.num_heads,            # e.g. 8
        depth=args.depth,                    # e.g. 2
        dropout=args.dropout,                # e.g. 0.1 or 0.05
        use_logit_input=args.use_logit_input, # scores are probabilities -> logit space
        init_output_bias=-4.0,
        base_weights=args.teacher_weights,
        num_decode_queries=args.num_decode_queries,
        # go_embed_dim=num_classes,
        # use_term_gate=args.use_term_gate,
        # gate_scale_init=args.gate_scale_init,
    ).to(device)
    ema = ModelEma(model, decay=0.997)

    train_go_emb = np.load(args.train_go_features).astype(np.float32)          # (C, Dg)
    valid_go_emb = np.load(args.valid_go_features).astype(np.float32)
    train_go_emb = torch.from_numpy(train_go_emb)
    valid_go_emb = torch.from_numpy(valid_go_emb)
    # model.set_go_embed(train_go_emb)

    loss_kwargs = dict(
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.clip,
        eps=args.eps,
    )
    if args.loss == "asl_opt":
        loss_kwargs["disable_torch_grad_focal_loss"] = args.disable_torch_grad_focal_loss

    criterion = create_loss(args.loss, **loss_kwargs)

    optimizer = build_optimizer(model, args.optimizer, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler(enabled=args.amp)
    best_val_fmax = -1.0
    best_val_aupr = -1.0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        model.set_go_embed(train_go_emb)
        for batch in train_loader:
            if len(batch) == 3:
                x, y, e = batch
            else:
                x, y = batch
                e = None

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if e is not None:
                e = e.to(device, non_blocking=True)

            if args.mixup_p > 0 and np.random.rand() < args.mixup_p:
                x, y, e = mixup_batch(x, y, e=e,alpha=args.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)


            with torch.amp.autocast(enabled=args.amp, device_type=device.type):
                logits_s, _ = model(x, e)
                loss_main = criterion(logits_s, y)

                loss_kd = torch.zeros((), device=device)
                if args.use_kd and args.teacher_weights is not None:
                    # args.teacher_weights: [w_zhao, w_esm2, w_profunsom]
                    with torch.no_grad():
                        w = torch.tensor(
                            args.teacher_weights,
                            device=x.device,
                            dtype=x.dtype,
                        ).view(1, 3, 1)              # (1,3,1)

                        # 如果 x 是概率，则直接加权；如果是 logits，需要先 sigmoid 一下
                        teacher_probs = (w * x).sum(dim=1)    # (B,C)
                        teacher_probs = teacher_probs.clamp(1e-6, 1 - 1e-6)

                    loss_kd = kd_loss_logits_masked(logits_s, teacher_probs)
                if args.use_ema_kd:
                    with torch.no_grad():
                        logits_t, _ = ema.module(x, e)   # teacher = EMA(model)

                    T = args.kd_T  # e.g. 2.0
                    # teacher prob，用 sigmoid + 温度，避免 logit 尖锐
                    p_t = torch.sigmoid(logits_t / T)
                    # student 用 logits 版本的 BCE，和你刚才修过的一致
                    loss_kd = F.binary_cross_entropy_with_logits(
                        logits_s / T, p_t
                    ) * (T * T)

                    # 可选：只对中置信度 term 做 KD 掩码
                    # mask = (p_t > 0.05) & (p_t < 0.7)
                    # loss_kd = F.binary_cross_entropy_with_logits(
                    #     logits_s[mask] / T, p_t[mask]
                    # ) * (T * T)

                loss = loss_main + args.kd_lambda * loss_kd

            # --- term gate regularization (keep per-term weights close to global weights) ---
            if getattr(model, "use_term_gate", False) and hasattr(model, "term_gate_logits") and args.gate_reg_lambda > 0:
                w_term = torch.softmax(model.term_gate_logits * model.gate_scale, dim=1)  # (1,3,C)
                # anchor = global base weights expanded to (1,3,C)
                w0 = model._get_base_weights().repeat(1, 1, model.num_classes)            # (1,3,C)
                gate_reg = (w_term - w0).pow(2).mean()
                loss = loss + (args.gate_reg_lambda * gate_reg)

            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf loss detected")
                print("logits min/max:", logits_s.min().item(), logits_s.max().item())
                print("x min/max:", x.min().item(), x.max().item())
                # optionally: check parameters
                for n, p in model.named_parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        print("Bad grad:", n)
                        break
                raise RuntimeError("Stop due to NaN/Inf")

            scaler.scale(loss).backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))

        model.set_go_embed(valid_go_emb)
        if (epoch % args.val_every) == 0:
            val_metrics = run_validation(model, val_loader, device, max_batches=args.val_max_batches)
            ema_val_metrics = run_validation(ema.module, val_loader, device, max_batches=args.val_max_batches)
            val_fmax = max(val_metrics["fmax"], ema_val_metrics["fmax"])
            val_aupr = max(val_metrics["aupr"], ema_val_metrics["aupr"])

            print(
                f"[Epoch {epoch:03d}/{args.epochs}] "
                f"train_loss={avg_loss:.6f} val_Fmax={val_fmax:.6f} val_AUPR={val_aupr:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.3e}"
            )

            if val_fmax > best_val_fmax and val_aupr > best_val_aupr:
                best_val_fmax = val_fmax
                best_val_aupr = val_aupr
                ckpt_path = os.path.join(args.save_dir, f"best_{args.go_name}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "ema_model": ema.module.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val_fmax": best_val_fmax,
                        "args": vars(args),
                        "num_classes": num_classes,
                    },
                    ckpt_path,
                )
                print(f"  Saved best checkpoint to: {ckpt_path}")
        else:
            print(f"[Epoch {epoch:03d}/{args.epochs}] train_loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.3e}")

    last_path = os.path.join(args.save_dir, f"last_{args.go_name}.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "ema_model": ema.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_fmax": best_val_fmax,
            "args": vars(args),
            "num_classes": num_classes,
        },
        last_path,
    )
    print(f"Training finished. Best val Fmax={best_val_fmax:.6f}. Saved last checkpoint to: {last_path}")

if __name__ == "__main__":
    main()