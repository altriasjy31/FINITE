#!/usr/bin/env python3
"""
Convert union-space NPZ predictions to NPY for faster training IO.

Input directory (example):
  ./results/union_space_preds_only/bp/train/
    - target_preds_union.npz   (keys: zhao, esm2_v3, profunsom) each (N, C)
    - union_labels.npy         (N, C) binary
    - union_go_terms.npy       (C,)
    - common_proteins.npy      (N,)

Outputs (written into --out_dir, default: same split_dir):
  - scores_union.npy           (N, 3, C) float16/float32 (C-order)
  - labels_union.npy           (N, C) bool or uint8
  - union_go_terms.npy         copied (optional)
  - common_proteins.npy        copied (optional)

This format supports np.load(..., mmap_mode="r") for fast row-wise batch slicing.
"""

import os
import shutil
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Convert union NPZ predictions to NPY.")
    p.add_argument("--split_dir", required=True, help="Path to a split folder (e.g., bp/train).")
    p.add_argument("--out_dir", default=None, help="Output folder. Default: same as split_dir.")
    p.add_argument("--scores_dtype", default="float16", choices=["float16", "float32"],
                   help="Data type for scores_union.npy.")
    p.add_argument("--labels_dtype", default="bool", choices=["bool", "uint8"],
                   help="Data type for labels_union.npy.")
    p.add_argument("--copy_meta", action="store_true",
                   help="Copy common_proteins.npy and union_go_terms.npy to out_dir.")
    return p.parse_args()


def main():
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else split_dir
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(split_dir, "target_preds_union.npz")
    labels_path = os.path.join(split_dir, "union_labels.npy")
    proteins_path = os.path.join(split_dir, "common_proteins.npy")
    go_terms_path = os.path.join(split_dir, "union_go_terms.npy")

    for fp in [npz_path, labels_path, proteins_path, go_terms_path]:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing file: {fp}")

    # --- Load NPZ predictions ---
    preds = np.load(npz_path, allow_pickle=False)
    required = ["zhao", "esm2_v3", "profunsom"]
    for k in required:
        if k not in preds:
            raise KeyError(f"Key '{k}' not found in {npz_path}. Available: {list(preds.keys())}")

    s1 = preds["zhao"]
    s2 = preds["esm2_v3"]
    s3 = preds["profunsom"]

    if not (s1.shape == s2.shape == s3.shape):
        raise ValueError(f"Shape mismatch: zhao={s1.shape}, esm2_v3={s2.shape}, profunsom={s3.shape}")

    n, c = s1.shape

    # --- Load labels ---
    y = np.load(labels_path, allow_pickle=False)
    if y.shape != (n, c):
        raise ValueError(f"Label shape mismatch: labels={y.shape}, scores={(n, c)}")

    # --- Stack scores to (N, 3, C) ---
    scores_dtype = np.float16 if args.scores_dtype == "float16" else np.float32
    scores = np.stack([s1, s2, s3], axis=1).astype(scores_dtype, copy=False)  # (N,3,C)

    # Ensure C-order for fast row slicing
    scores = np.ascontiguousarray(scores)

    # --- Convert labels dtype ---
    if args.labels_dtype == "bool":
        labels = (y > 0).astype(np.bool_, copy=False)
    else:
        labels = (y > 0).astype(np.uint8, copy=False)
    labels = np.ascontiguousarray(labels)  # C-order

    # --- Save ---
    out_scores = os.path.join(out_dir, "scores_union.npy")
    out_labels = os.path.join(out_dir, "labels_union.npy")
    np.save(out_scores, scores)
    np.save(out_labels, labels)

    # --- Optionally copy metadata ---
    if args.copy_meta:
        shutil.copy2(proteins_path, os.path.join(out_dir, "common_proteins.npy"))
        shutil.copy2(go_terms_path, os.path.join(out_dir, "union_go_terms.npy"))

    print(f"[OK] Converted: {split_dir}")
    print(f"  scores_union.npy: {out_scores}  shape={scores.shape} dtype={scores.dtype}")
    print(f"  labels_union.npy: {out_labels}  shape={labels.shape} dtype={labels.dtype}")
    if args.copy_meta:
        print("  Metadata copied: common_proteins.npy, union_go_terms.npy")


if __name__ == "__main__":
    main()