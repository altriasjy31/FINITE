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
from util.metrics import fmax_score


# -----------------------------
# 工具函数：加载 ESM2 特征 & node.dat 名称
# -----------------------------
def load_esm2_features(go_name: str, device: torch.device):
    """
    按照当前 trainer 中 load_esm2_features 的路径组织：
    /data0/shaojiangyi/pprogo-flg-2/data/{go_name}/esm2_features/*.npy
    """
    esm2_dir = P.Path(f"/data0/shaojiangyi/pprogo-flg-2/data/{go_name}/esm2_features")
    protein_path = esm2_dir / "esm2_protein_features.npy"
    go_path = esm2_dir / "esm2_go_features.npy"

    esm2_protein = None
    esm2_go = None

    if protein_path.exists():
        arr = np.load(protein_path)
        esm2_protein = torch.as_tensor(arr, dtype=torch.float32, device=device)
        print(f"[INFO] ESM2 protein feats: {protein_path} shape={tuple(esm2_protein.shape)}")
    else:
        print(f"[WARN] Missing ESM2 protein feats: {protein_path}")

    if go_path.exists():
        arr = np.load(go_path)
        esm2_go = torch.as_tensor(arr, dtype=torch.float32, device=device)
        print(f"[INFO] ESM2 GO feats: {go_path} shape={tuple(esm2_go.shape)}")
    else:
        print(f"[WARN] Missing ESM2 GO feats: {go_path}")

    return esm2_protein, esm2_go


def load_node_lists(go_name: str) -> Tuple[List[str], List[str]]:
    """
    从 node.dat 读取蛋白质名称 & GO 术语名称（与 trainer 的 load_node_names 保持一致）

    Return:
      protein_names_all: list[str], length = #protein (按照 full graph protein id 顺序)
      go_terms_all:      list[str], length = #go      (按照 full graph go id 顺序)
    """
    node_path = P.Path(f"/data0/shaojiangyi/pprogo-flg-2/data/{go_name}/node.dat")
    node = pd.read_csv(node_path, header=None, sep="\t", dtype={3: str})
    node.columns = ["id", "name", "type", "feature"]

    protein_nodes = node[node["type"] == 0].reset_index(drop=True)
    go_nodes = node[node["type"] == 1].reset_index(drop=True)

    protein_names_all = protein_nodes["name"].tolist()
    go_terms_all = go_nodes["name"].tolist()

    print(f"[INFO] node.dat: proteins={len(protein_names_all)} go_terms={len(go_terms_all)}")
    return protein_names_all, go_terms_all


# -----------------------------
# 预测逻辑（与 trainer.evaluate / save_predictions 对齐）
# -----------------------------
@torch.no_grad()
def predict_split_with_dataset(
    model: nn.Module,
    dataset: DBLPDataset,
    split: str,
    protein_names_all: List[str],
    esm2_protein: Optional[torch.Tensor],
    esm2_go: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
    amp: bool = True,
    need_eval: bool = True,
):
    """
    在给定 split（train/valid/test）上，用最新的 DBLPDataset / trainer 逻辑做预测。

    返回：
      protein_parent_ids: np.ndarray[int], full graph 蛋白质 ID
      protein_names:      np.ndarray[str], 蛋白质名称（node.dat 对应）
      y_pred:             np.ndarray[float32], (N, C) 预测概率
      y_true:             np.ndarray[float32], (N, C) 多标签真实标记
    """
    split = split.lower()

    if split == "train":
        base_g = dataset.train_g
        seeds_local = dataset.train_seeds
        split_parent_idx = dataset.train_idx  # full graph protein ids (tensor)
    elif split in ["valid", "val"]:
        base_g = dataset.valid_g
        seeds_local = dataset.valid_seeds
        split_parent_idx = dataset.valid_idx
    elif split == "test":
        base_g = dataset.test_g
        seeds_local = dataset.test_seeds
        split_parent_idx = dataset.test_idx
    else:
        raise ValueError(f"Unknown split: {split}")

    # 使用 dataset.sampler，保证与训练/验证时一致
    sampler = dataset.sampler

    loader = dgl.dataloading.DataLoader(
        base_g,
        {"protein": seeds_local},
        sampler,
        batch_size=batch_size,
        shuffle=False,
        device=device,
        drop_last=False,
        num_workers=0,
    )

    all_probs = []
    all_labels = []
    all_parent_ids = []

    model.eval()

    # 可能存在 parent_nid，也可能只有 dgl.NID（按你是否已经更新 dataset 而定）
    def _get_parent_ids_from_sg(sg: dgl.DGLHeteroGraph, ntype: str):
        data = sg.nodes[ntype].data
        if "parent_nid" in data:
            return data["parent_nid"]
        elif dgl.NID in data:
            return data[dgl.NID]
        else:
            raise KeyError(f"Neither 'parent_nid' nor dgl.NID found in sg.nodes[{ntype}].data")

    def _get_parent_ids_from_base(base_g: dgl.DGLHeteroGraph, ntype: str, local_ids: torch.Tensor):
        data = base_g.nodes[ntype].data
        if "parent_nid" in data:
            return data["parent_nid"][local_ids]
        elif dgl.NID in data:
            return data[dgl.NID][local_ids]
        else:
            raise KeyError(f"Neither 'parent_nid' nor dgl.NID found in base_g.nodes[{ntype}].data")

    model.eval()
    all_preds = []
    all_ema_preds = []
    all_labels = []

    n_batches = 0

    if isinstance(device, torch.device):
        amp = device.type == "cuda" and torch.cuda.is_available()
    elif isinstance(device, str):
        amp = device.startswith("cuda") and torch.cuda.is_available()
    else:
        raise ValueError(f"Unknown device type: {type(device)}")

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in loader:
            # 只关心有 protein 输出的 batch
            if "protein" not in output_nodes:
                continue

            # 1) 构造 batch 子图
            sg = dgl.node_subgraph(base_g, input_nodes)

            # 2) MSA / ProFun-SOM 特征
            h_msa_dict = sg.ndata["h"]

            # 3) ESM2 特征
            esm2_dict = {}
            if esm2_protein is not None and sg.num_nodes("protein") > 0:
                p_parent = sg.nodes["protein"].data["parent_nid"]
                esm2_dict["protein"] = esm2_protein[p_parent]

            if esm2_go is not None and sg.num_nodes("go_annotation") > 0:
                g_parent = sg.nodes["go_annotation"].data["parent_nid"]
                esm2_dict["go_annotation"] = esm2_go[g_parent]

            # 4) 前向：得到子图所有 protein 的 logits
            if amp:
                with autocast(device_type=device.type, enabled=True):
                    out = model(sg, h_msa_dict, esm2_dict)
            else:
                out = model(sg, h_msa_dict, esm2_dict)

            logits_all = out["protein"]  # (N_protein_in_sg, num_classes)

            # 5) 将 output_nodes['protein'] (base_g id) 映射到 sg 的 local index
            out_p = output_nodes["protein"]    # in base_g id space
            sg_base_ids = sg.nodes["protein"].data[dgl.NID].detach().cpu().tolist()
            id2idx = {int(pid): i for i, pid in enumerate(sg_base_ids)}

            try:
                out_p_list = out_p.detach().cpu().tolist()
                idx_list = [id2idx[int(pid)] for pid in out_p_list]
            except KeyError:
                # 极少出现不一致时，跳过这个 batch
                continue

            idx_tensor = torch.tensor(idx_list, device=device, dtype=torch.long)
            logits = logits_all[idx_tensor]  # (B, num_classes)

            # 6) labels：通过 base_g.parent_nid -> full graph id -> get_label
            parent_ids = base_g.nodes["protein"].data["parent_nid"][out_p]
            parent_ids_list = parent_ids.detach().cpu().tolist()
            # labels_np = self.dataset.get_label(parent_ids_list)
            labels_np = dataset.get_label_eval(parent_ids_list)
            labels = torch.tensor(labels_np, dtype=torch.float32, device=device)

            if logits.ndim != 2 or logits.shape != labels.shape:
                raise RuntimeError(
                    f"[eval shape mismatch] logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
                )

            n_batches += 1

            probs = torch.sigmoid(logits)
            all_parent_ids.append(np.asarray(parent_ids_list, dtype=np.int64))
            all_preds.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    protein_parent_ids = np.concatenate(all_parent_ids, axis=0)  # full graph ids
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_ema_preds = np.concatenate(all_ema_preds, axis=0) if len(all_ema_preds) > 0 else None

    if need_eval:
        fmax, aupr = fmax_score(all_labels, all_preds, auprc=True)
        print(f"[EVAL] split={split} fmax={fmax:.4f} auprc={aupr:.4f} samples={all_labels.shape[0]}")

    # ids -> names (node.dat protein list 按 full graph id 对齐)
    protein_names = np.asarray(
        [protein_names_all[int(pid)] for pid in protein_parent_ids],
        dtype=object,
    )

    return protein_parent_ids, protein_names, all_preds, all_labels, split_parent_idx.detach().cpu().numpy()


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go_name", type=str, required=True, choices=["bp", "cc", "mf"])
    parser.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"])
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP during prediction")
    parser.add_argument("--no_eval", action="store_true", help="Do not compute evaluation metrics")
    parser.add_argument("--no_ema", action="store_true", help="Do not use EMA weights even if available")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    amp = not args.no_amp
    ema_flag = not args.no_ema
    need_eval = not args.no_eval

    # 1) 构建 dataset（与训练一致）
    dataset = DBLPDataset('data', args.go_name, 'HGAT', args.batch_size, device) 

    # 2) 加载 node.dat 中的蛋白质与 GO 名称
    protein_names_all, go_terms_all = load_node_lists(args.go_name)

    # 3) 加载 ESM2 特征（与 trainer 的路径保持一致）
    esm2_protein, esm2_go = load_esm2_features(args.go_name, device)

    # 4) 构建模型（超参数与训练时保持一致）
    model = HGAT_ESM2_V3(
                dataset.node_type,
                num_classes=dataset.go_num,
                protein_feat_dim=dataset.feature_dim,
                go_feat_dim=getattr(dataset, 'go_feature_dim', dataset.feature_dim),
                hidden_dim=getattr(args, 'hidden_dim', 256),
                num_layers=getattr(args, 'num_layers', 2),
                use_esm2=True,
                esm2_protein_dim=1280,
                esm2_go_dim=1280,
            ).to(device)

    # 5) 加载训练好参数
    ckpt = torch.load(args.model_path, map_location=device)
    if ema_flag:
        state = ckpt.get("ema_model", None)
    else:
        state = ckpt.get("model_state_dict", None)
    
    assert state is not None, "No model state dict found in checkpoint"
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[INFO] Loaded model from {args.model_path}")

    # 6) 执行预测（在 train/valid/test 对应子图上）
    protein_ids, protein_names, y_pred, y_true, target_parent_idx = predict_split_with_dataset(
        model=model,
        dataset=dataset,
        split=args.split,
        protein_names_all=protein_names_all,
        esm2_protein=esm2_protein,
        esm2_go=esm2_go,
        batch_size=args.batch_size,
        device=device,
        amp=amp,
        need_eval=need_eval,
    )

    # 7) 覆盖率 sanity check
    target_set = set(target_parent_idx.tolist())
    pred_set = set(protein_ids.tolist())
    missing = sorted(list(target_set - pred_set))
    extra = sorted(list(pred_set - target_set))

    print(f"[CHECK] split={args.split} target={len(target_set)} predicted={len(pred_set)} "
          f"missing={len(missing)} extra={len(extra)}")
    if missing:
        print(f"[CHECK] missing head: {missing[:20]}")

    # 8) 保存结果
    out_path = P.Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        protein_names=protein_names,
        protein_ids=protein_ids,
        predictions=y_pred,
        labels=y_true,
        go_terms=np.asarray(go_terms_all),
        split=args.split,
    )
    print(f"[INFO] Saved: {out_path} | preds={y_pred.shape} proteins={len(protein_names)}")


if __name__ == "__main__":
    main()