#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HGAT prediction script.

- Loads a trained HGAT checkpoint.
- Runs inference on dataset.test_loader (or other loader if you modify).
- Saves predictions to a user-specified output path.

Output format:
1) .npz (recommended): protein_ids, y_pred (float32), (optional) y_true
2) .csv (optional): protein_id + per-GO probabilities
"""

import os
import sys
import argparse
import pathlib as P
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
import dgl
from tqdm import tqdm

sys.path.append('..')
prj_root = str(P.Path(__file__).parent.parent)
if prj_root not in sys.path:
    sys.path.append(prj_root)

from models.nnHGAT_esm2_v3 import HGAT_ESM2_V3
from dataset.inductive_learning_dataset import DBLPDataset


def load_esm2_features(go_name: str, device: torch.device):
    esm2_dir = P.Path(f"/data0/lmj/pprogo-flg/data/{go_name}/esm2_features")
    protein_path = esm2_dir / "esm2_protein_features.npy"
    go_path = esm2_dir / "esm2_go_features.npy"

    esm2_protein = None
    esm2_go = None
    if protein_path.exists():
        esm2_protein = torch.tensor(np.load(protein_path), dtype=torch.float32, device=device)
        print(f"[INFO] ESM2 protein feats: {protein_path} | {tuple(esm2_protein.shape)}")
    else:
        print(f"[WARN] Missing ESM2 protein feats: {protein_path}")

    if go_path.exists():
        esm2_go = torch.tensor(np.load(go_path), dtype=torch.float32, device=device)
        print(f"[INFO] ESM2 GO feats: {go_path} | {tuple(esm2_go.shape)}")
    else:
        print(f"[WARN] Missing ESM2 GO feats: {go_path}")

    return esm2_protein, esm2_go


def load_node_lists(go_name: str) -> Tuple[List[str], List[str]]:
    """
    Return:
      protein_names_in_graph_order: list[str] length = #protein nodes (type==0)
      go_terms_in_graph_order:      list[str] length = #go nodes (type==1)
    """
    node_path = P.Path(f"/data0/lmj/pprogo-flg/data/{go_name}/node.dat")
    node = pd.read_csv(node_path, header=None, sep="\t", dtype={3: str})
    protein_names = node[node[2] == 0][1].tolist()
    go_terms = node[node[2] == 1][1].tolist()
    print(f"[INFO] node.dat: proteins={len(protein_names)} go_terms={len(go_terms)}")
    return protein_names, go_terms


@torch.no_grad()
def predict_split_on_parent_graph(
    model: nn.Module,
    hg: dgl.DGLHeteroGraph,
    dataset: DBLPDataset,
    target_idx: torch.Tensor,          # parent graph protein ids
    protein_names_all: List[str],      # node.dat protein names in graph order
    esm2_protein: Optional[torch.Tensor],
    esm2_go: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    amp: bool = True,
):
    sampler = dgl.dataloading.NeighborSampler([5])  # 和你训练保持一致即可；需要更深可改成 [5,5] 等

    # IMPORTANT: use parent graph hg and parent-ids seeds
    loader = dgl.dataloading.DataLoader(
        hg,
        {"protein": target_idx},
        sampler,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        drop_last=False,
        num_workers=0,
    )

    all_probs = []
    all_ids = []
    all_labels = []

    model.eval()

    for input_nodes, output_nodes, blocks in tqdm(loader, ncols=120, desc="Predict"):
        # features from blocks[0].srcdata['h']
        h_dict = {ntype: blocks[0].srcdata["h"][ntype] for ntype in blocks[0].ntypes}

        # optional esm2 feature gather (by input_nodes ids in parent graph)
        esm2_p = None
        esm2_g = None
        if esm2_protein is not None and "protein" in input_nodes:
            esm2_p = esm2_protein[input_nodes["protein"]]
        if esm2_go is not None and "go_annotation" in input_nodes:
            esm2_g = esm2_go[input_nodes["go_annotation"]]

        with autocast(enabled=amp, device_type="cuda" if device.type == "cuda" else "cpu"):
            out = model(blocks[-1], h_dict, esm2_p, esm2_g)
            logits_all = out["protein"]  # logits for dst proteins in last block

        # output_nodes['protein'] are parent ids (because loader is built on hg with parent seeds)
        out_parent_ids = output_nodes["protein"]

        # map dst parent ids -> row indices in logits_all
        dst_parent_ids = blocks[-1].dstnodes["protein"].data[dgl.NID]
        nid_to_row = {int(n.item()): i for i, n in enumerate(dst_parent_ids)}

        rows = []
        keep_ids = []
        for nid in out_parent_ids.detach().cpu().numpy().tolist():
            if int(nid) in nid_to_row:
                keep_ids.append(int(nid))
                rows.append(nid_to_row[int(nid)])

        if len(rows) == 0:
            continue

        rows_t = torch.tensor(rows, device=logits_all.device, dtype=torch.long)
        probs = torch.sigmoid(logits_all[rows_t]).to(torch.float32)

        labels_list = dataset.get_label(keep_ids)   # List[List[int]] or List[List[0/1]]
        labels_np = np.asarray(labels_list, dtype=np.float32)  # (B, C

        all_ids.append(np.asarray(keep_ids, dtype=np.int64))
        all_probs.append(probs.detach().cpu().numpy().astype(np.float32))
        all_labels.append(labels_np) 

        del h_dict, out, logits_all, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    protein_ids = np.concatenate(all_ids, axis=0)
    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # ids -> names (node.dat protein list indexed by parent id)
    protein_names = np.asarray([protein_names_all[int(pid)] for pid in protein_ids], dtype=object)

    return protein_ids, protein_names, y_pred, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go_name", type=str, required=True, choices=["bp", "cc", "mf"])
    parser.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    # build dataset (graph + features)
    dataset = DBLPDataset("data", args.go_name, "HGAT", args.batch_size, device)

    # OVERRIDE split by list files
    dataset.get_split_from_lists()

    # target idx from list files
    if args.split == "train":
        target_idx = dataset.train_idx
    elif args.split == "valid":
        target_idx = dataset.valid_idx
    else:
        target_idx = dataset.test_idx

    # load node names
    protein_names_all, go_terms = load_node_lists(args.go_name)

    # load esm2 feats (optional)
    esm2_protein, esm2_go = load_esm2_features(args.go_name, device)

    # build model
    model = HGAT_ESM2_V3(
        dataset.node_type,
        num_classes=dataset.go_num,
        feature_dim=dataset.feature_dim,
        hidden_dim=64,
        num_layers=2,
        go_feature_dim=getattr(dataset, "go_feature_dim", None),
        use_esm2=True,
        esm2_protein_dim=1280,
        esm2_go_dim=1280,
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # IMPORTANT: use parent graph
    hg = dataset.g.to(device)

    protein_ids, protein_names, y_pred, y_true = predict_split_on_parent_graph(
        model=model,
        hg=hg,
        dataset=dataset,
        target_idx=target_idx,
        protein_names_all=protein_names_all,
        esm2_protein=esm2_protein,
        esm2_go=esm2_go,
        batch_size=args.batch_size,
        device=device,
        amp=True,
    )

    # sanity check: coverage
    target_set = set(target_idx.detach().cpu().numpy().tolist())
    pred_set = set(protein_ids.tolist())
    missing = sorted(list(target_set - pred_set))
    extra = sorted(list(pred_set - target_set))
    print(f"[CHECK] target={len(target_set)} predicted={len(pred_set)} missing={len(missing)} extra={len(extra)}")
    if missing:
        print(f"[CHECK] missing head: {missing[:20]}")

    out_path = P.Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        protein_names=protein_names,
        protein_ids=protein_ids,
        predictions=y_pred,
        labels=y_true,
        go_terms=np.asarray(go_terms),
        split=args.split,
    )
    print(f"[INFO] Saved: {out_path} | preds={y_pred.shape} proteins={len(protein_names)}")


if __name__ == "__main__":
    main()