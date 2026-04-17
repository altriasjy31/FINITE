#!/usr/bin/env python3
"""
Inference for fusion model on a single HDF5 file.

Outputs:
  - probs.npy: (N, C) float32 probabilities
  - gate_weights.npy: (N, 3) float32 channel weights
Optionally prints metrics if labels are available in HDF5.
"""

import os
import argparse
import pathlib as P
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse your model + dataset utils
# Ensure these are importable from your training codebase.
from train_fusion_project_then_fuse import (FusionProjectThenFuse, 
                                            FusionTransformerLatent, 
                                            FusionLinearBaseTransformerResidual,
                                            NGHFusion)
from train_fusion_project_then_fuse import H5IndexDataset
from train_fusion_project_then_fuse import run_validation
import h5py

class LogitsOnly(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        logits, _ = self.m(x)
        return logits


def parse_args():
    p = argparse.ArgumentParser("Infer fusion model on HDF5.")
    p.add_argument("--num_heads", type=int, default=8)

    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--latent_tokens", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--res_scale", type=float, default=0.5)
    p.add_argument("--test_go_features", type=str)
    p.add_argument("--test_prot_features", type=str)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--use_logit_input", dest="use_logit_input", action="store_true",
                help="Treat input scores as probabilities and convert to logits.")
    g.add_argument("--no_logit_input", dest="use_logit_input", action="store_false",
                help="Treat input scores as logits already.")
    p.set_defaults(use_logit_input=None)  # None means 'not specified by CLI'

    p.add_argument("--use_term_gate", action="store_true")
    p.add_argument("--no_term_gate", dest="use_term_gate", action="store_false")
    p.set_defaults(use_term_gate=None)  # None means: use ckpt

    p.add_argument(
        "--teacher_weights", nargs=3, type=float, default=None,
        help="Base fusion weights (zhao, esm2_v3, profunsom)."
    )

    p.add_argument("--num_decode_queries", type=int)

    p.add_argument("--backend", type=str, default="h5", choices=["h5"])
    p.add_argument("--go_name", type=str, required=True, choices=["bp", "cc", "mf"])
    p.add_argument("--train_h5", type=str, required=True, help="HDF5 file path used in training (for go_terms).")
    p.add_argument("--h5", type=str, required=True, help="Input HDF5 file path.")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt path.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")

    return p.parse_args()


@torch.no_grad()
def infer(model, loader, device, amp=False):
    model.eval()
    all_probs = []

    # Aux collections (optional)
    cross_attn_list = []      # (B,K,3) if available

    for batch in loader:
        if len(batch) == 3:
            x, y, e = batch
        else:
            x, y = batch
            e = None
        x = x.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)

        with torch.amp.autocast(enabled=amp, device_type=device.type):
            logits, aux = model(x, e)  # aux is a dict
            probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu().numpy())

        # Save optional aux
        if isinstance(aux, dict):
            if "cross_attn_weights" in aux and torch.is_tensor(aux["cross_attn_weights"]):
                # Typically (B,K,3) or (B,3,K) depending on MHA; we keep as-is
                cross_attn_list.append(aux["cross_attn_weights"].detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)

    out_aux = {}
    if len(cross_attn_list) > 0:
        out_aux["cross_attn_weights"] = np.concatenate(cross_attn_list, axis=0)

    return probs, out_aux



def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    def load_go_terms(h5_path):
        with h5py.File(h5_path, "r") as f:
            return np.array(f["go_terms"]).astype(str)

    # Example: you need to pass train_h5 path
    train_terms = load_go_terms(args.train_h5)
    test_terms = load_go_terms(args.h5)
    print("[Check] go_terms equal:", np.array_equal(train_terms, test_terms))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = H5IndexDataset(args.h5, args.test_prot_features)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=ds.collate_fn,
        drop_last=False,
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", ds.c))

    saved_args = ckpt.get("args", {}) or {}

    def _require_same(name, cli_val, ckpt_val):
        if cli_val is None:
            return ckpt_val
        if bool(cli_val) != bool(ckpt_val):
            raise ValueError(f"[ConfigMismatch] {name}: CLI={cli_val} CKPT={ckpt_val}")
        return cli_val

    use_term_gate = _require_same("use_term_gate", args.use_term_gate, saved_args.get("use_term_gate", True))
    gate_scale_init = float(saved_args.get("gate_scale_init", 1.0))  # init not critical at infer but keep for completeness

    use_logit_input = _require_same("use_logit_input", args.use_logit_input, saved_args.get("use_logit_input", True))


    # Always rebuild model from checkpoint args first (CLI only as fallback).
    hidden_dim = int(saved_args.get("hidden_dim", args.hidden_dim))
    dropout = float(saved_args.get("dropout", args.dropout))
    num_heads = int(saved_args.get("num_heads", args.num_heads))
    depth = int(saved_args.get("depth", args.depth))
    latent_tokens = int(saved_args.get("latent_tokens", args.latent_tokens))

    # IMPORTANT: must match training
    use_logit_input = bool(saved_args.get("use_logit_input", True))

    # state = ckpt["model_state"]
    if "ema_model" in ckpt:
        print("[Info] Using EMA model for inference.")
        state = ckpt["ema_model"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        raise KeyError("Checkpoint missing 'model_state' or 'ema_model'.")

    model = NGHFusion(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        latent_tokens=latent_tokens,
        num_heads=num_heads,
        depth=depth,
        dropout=dropout,
        use_logit_input=use_logit_input,
        init_output_bias=float(saved_args.get("init_output_bias", -4.0)),
        base_weights=args.teacher_weights,
        num_decode_queries=args.num_decode_queries
        # go_embed_dim=num_classes,
        # use_term_gate=use_term_gate,
        # gate_scale_init=gate_scale_init,
    ).to(device)

    test_go_emb = np.load(args.test_go_features).astype(np.float32)          # (C, Dg)
    test_go_emb = torch.from_numpy(test_go_emb)
    model.set_go_embed(test_go_emb)

    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print("[Warn] load_state_dict strict=False")
        if missing:
            print("  missing keys:")
            for k in missing:
                print("   -", k)
        if unexpected:
            print("  unexpected keys:")
            for k in unexpected:
                print("   -", k)


    probs, aux = infer(model, loader, device, amp=args.amp)

    np.save(os.path.join(args.out_dir, f"{args.go_name}_test_probs.npy"),
            probs.astype(np.float32, copy=False))


    if "cross_attn_weights" in aux:
        np.save(os.path.join(args.out_dir, f"{args.go_name}_cross_attn.npy"),
                aux["cross_attn_weights"].astype(np.float32, copy=False))

    with torch.no_grad():
        batch = next(iter(loader))
        if len(batch) == 3:
            x0, y0, e0 = batch
        else:
            x0, y0 = batch
            e0 = None
        x0 = x0.to(device)
        e0 = e0.to(device)
        logits0, aux0 = model(x0, e0)
        if "term_gate_mean" in aux0:
            print("[Info] term_gate_mean (avg over terms):", aux0["term_gate_mean"].cpu().numpy())

    # Optional: compute metrics if labels exist in HDF5
    try:
        metrics = run_validation(model, loader, device, max_batches=None)
        print(f"[Metrics on {args.h5}] Fmax={metrics['fmax']:.6f} AUPR={metrics['aupr']:.6f}")
    except Exception as e:
        print(f"[Info] Skip metric computation: {e}")

    print(f"[OK] Saved:")
    print(f"  - {args.go_name}_test_probs.npy")
    print(f"  - {args.go_name}_cross_attn.npy (if available)")



if __name__ == "__main__":
    main()
