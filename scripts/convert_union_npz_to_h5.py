#!/usr/bin/env python3
"""
Pack union-space data into a single HDF5 file.

It can read either:
  - Original split_dir with target_preds_union.npz + union_labels.npy, OR
  - Already converted scores_union.npy + labels_union.npy

Outputs:
  - dataset.h5 with datasets:
      /scores   (N,3,C) float16/float32
      /labels   (N,C)   uint8 or bool
      /proteins (N,)    variable-length string
      /go_terms (C,)    variable-length string

Chunking is tuned for row-wise batch reads: (chunk_rows, 3, C).
"""

import os
import argparse
import numpy as np
import h5py


def parse_args():
    p = argparse.ArgumentParser("Convert union data to HDF5.")
    p.add_argument("--split_dir", required=True, help="Path to split folder.")
    p.add_argument("--out_h5", required=True, help="Output HDF5 file path.")
    p.add_argument("--use_npy", action="store_true",
                   help="If set, read scores_union.npy and labels_union.npy instead of NPZ+union_labels.npy.")
    p.add_argument("--scores_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--labels_dtype", default="uint8", choices=["uint8", "bool"])
    p.add_argument("--chunk_rows", type=int, default=64,
                   help="Row chunk size; set close to your training batch size.")
    p.add_argument("--compression", default=None, choices=[None, "gzip", "lzf"],
                   help="Compression can reduce disk use but may reduce throughput.")
    return p.parse_args()


def main():
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    out_h5 = os.path.abspath(args.out_h5)
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)

    proteins_path = os.path.join(split_dir, "common_proteins.npy")
    go_terms_path = os.path.join(split_dir, "union_go_terms.npy")
    if not os.path.isfile(proteins_path) or not os.path.isfile(go_terms_path):
        raise FileNotFoundError("Missing common_proteins.npy or union_go_terms.npy")

    proteins = np.load(proteins_path, allow_pickle=True)
    go_terms = np.load(go_terms_path, allow_pickle=True)

    if args.use_npy:
        scores_path = os.path.join(split_dir, "scores_union.npy")
        labels_path = os.path.join(split_dir, "labels_union.npy")
        if not os.path.isfile(scores_path) or not os.path.isfile(labels_path):
            raise FileNotFoundError("Missing scores_union.npy or labels_union.npy")
        scores = np.load(scores_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        # dtype coercion happens when writing into HDF5
    else:
        npz_path = os.path.join(split_dir, "target_preds_union.npz")
        labels_path = os.path.join(split_dir, "union_labels.npy")
        if not os.path.isfile(npz_path) or not os.path.isfile(labels_path):
            raise FileNotFoundError("Missing target_preds_union.npz or union_labels.npy")

        preds = np.load(npz_path, allow_pickle=False)
        s1 = preds["zhao"]
        s2 = preds["esm2_v3"]
        s3 = preds["profunsom"]
        scores = np.stack([s1, s2, s3], axis=1)
        labels = np.load(labels_path, allow_pickle=False)

    # Basic checks
    if scores.ndim != 3 or scores.shape[1] != 3:
        raise ValueError(f"scores must have shape (N,3,C), got {scores.shape}")
    n, _, c = scores.shape
    if labels.shape != (n, c):
        raise ValueError(f"labels shape mismatch: {labels.shape} vs {(n, c)}")
    if len(proteins) != n:
        raise ValueError(f"proteins length mismatch: {len(proteins)} vs N={n}")
    if len(go_terms) != c:
        raise ValueError(f"go_terms length mismatch: {len(go_terms)} vs C={c}")

    scores_dtype = np.float16 if args.scores_dtype == "float16" else np.float32
    labels_dtype = np.bool_ if args.labels_dtype == "bool" else np.uint8

    # Variable-length strings
    str_dt = h5py.string_dtype(encoding="utf-8")

    # Write HDF5
    with h5py.File(out_h5, "w") as f:
        # Datasets
        f.create_dataset(
            "scores",
            shape=(n, 3, c),
            dtype=scores_dtype,
            chunks=(args.chunk_rows, 3, c),
            compression=args.compression,
        )
        f.create_dataset(
            "labels",
            shape=(n, c),
            dtype=labels_dtype,
            chunks=(args.chunk_rows, c),
            compression=args.compression,
        )

        # Write in chunks to avoid huge peak memory
        for i in range(0, n, args.chunk_rows):
            j = min(i + args.chunk_rows, n)
            # f["scores"][i:j] = np.asarray(scores[i:j], dtype=scores_dtype)
            chunk = scores[i:j]

            # If list/tuple -> stack
            if isinstance(chunk, (list, tuple)):
                chunk = np.stack(chunk, axis=0)

            chunk = np.asarray(chunk)

            # Fix axis order if needed (common: (B,C,3) -> (B,3,C))
            if chunk.ndim == 3 and chunk.shape[1] != 3 and chunk.shape[2] == 3:
                chunk = np.transpose(chunk, (0, 2, 1))

            # Enforce expected shape
            if chunk.shape != (j - i, 3, c):
                raise ValueError(f"scores chunk shape mismatch: got {chunk.shape}, expected {(j-i,3,c)}")

            # Enforce numeric dtype (fails fast if strings are present)
            if chunk.dtype.kind in ("U", "S", "O"):
                if chunk.dtype.kind == "S":
                    chunk = np.char.decode(chunk, "utf-8")
                chunk = chunk.astype(np.float32)  # may raise if non-numeric tokens exist

            chunk = chunk.astype(scores_dtype, copy=False)
            f["scores"][i:j] = chunk
            if labels_dtype == np.bool_:
                f["labels"][i:j] = (np.asarray(labels[i:j]) > 0)
            else:
                f["labels"][i:j] = (np.asarray(labels[i:j]) > 0).astype(np.uint8)

        str_dt = h5py.string_dtype(encoding="utf-8")

        prot_ds = f.create_dataset("proteins", shape=(len(proteins),), dtype=str_dt)
        prot_ds[...] = proteins.astype(str).tolist()

        go_ds = f.create_dataset("go_terms", shape=(len(go_terms),), dtype=str_dt)
        go_ds[...] = go_terms.astype(str).tolist()

        print("proteins dtype:", proteins.dtype, "shape:", proteins.shape)
        print("go_terms dtype:", go_terms.dtype, "shape:", go_terms.shape)


        # Store small metadata
        f.attrs["N"] = n
        f.attrs["C"] = c

    print(f"[OK] Wrote HDF5: {out_h5}")
    print(f"  scores: (N,3,C)=({n},3,{c}) dtype={scores_dtype} chunks=({args.chunk_rows},3,{c})")
    print(f"  labels: (N,C)=({n},{c}) dtype={labels_dtype} chunks=({args.chunk_rows},{c})")


if __name__ == "__main__":
    main()
