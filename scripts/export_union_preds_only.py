#!/usr/bin/env python3

import argparse
import pathlib as P
import json
import numpy as np
import pickle
import pandas as pd

import sys
sys.path.insert(0, str(P.Path(__file__).parent))

from fuse_three_models import align_labels, align_proteins, evaluate_predictions

def align_labels(predicted_results, o1, o2, fill_value=0.0):
    """
    Align prediction results to match the new label order, handling missing labels.

    Args:
        predicted_results: numpy array, shape (N, len(o1))
        o1: list - original label order
        o2: list - target label order (can have different labels/length)
        fill_value: value used for labels in o2 that do not exist in o1

    Returns:
        numpy array, shape (N, len(o2)) - aligned prediction results
    """
    N = predicted_results.shape[0]
    M_new = len(o2)
    
    # 初始化新的结果数组
    aligned_results = np.full((N, M_new), fill_value, dtype=predicted_results.dtype)
    
    # 创建从标签到o1中列索引的映射
    o1_label_to_idx = {label: idx for idx, label in enumerate(o1)}
    
    # 填充对齐后的结果
    for new_idx, label in enumerate(o2):
        if label in o1_label_to_idx:
            old_idx = o1_label_to_idx[label]
            aligned_results[:, new_idx] = predicted_results[:, old_idx]
    
    return aligned_results

def align_proteins(protein_list_1, protein_list_2, protein_list_3):
    """对齐三个模型的蛋白质列表"""
    print(f"[步骤5] 对齐蛋白质列表...")
    
    # 转换为字符串集合以便匹配
    set1 = set(str(p) for p in protein_list_1)
    set2 = set(str(p) for p in protein_list_2)
    set3 = set(str(p) for p in protein_list_3)
    
    # 找到三个集合的交集
    common_proteins = sorted(list(set1 & set2 & set3))
    print(f"  共同蛋白质数量: {len(common_proteins)} (模型1: {len(set1)}, 模型2: {len(set2)}, 模型3: {len(set3)})")
    
    # 创建索引映射
    dict1 = {str(p): idx for idx, p in enumerate(protein_list_1)}
    dict2 = {str(p): idx for idx, p in enumerate(protein_list_2)}
    dict3 = {str(p): idx for idx, p in enumerate(protein_list_3)}
    
    indices1 = [dict1[p] for p in common_proteins]
    indices2 = [dict2[p] for p in common_proteins]
    indices3 = [dict3[p] for p in common_proteins]
    
    return common_proteins, indices1, indices2, indices3

def evaluate_predictions(y_true, y_pred):
    """
    评估预测结果（Fmax和AUPR）
    使用与ProFun-SOM相同的评估方法：全局precision/recall
    """
    # 过滤掉没有标签的蛋白质
    has_labels = np.sum(y_true, axis=1) > 0
    if np.sum(has_labels) == 0:
        return 0.0, 0.0
    
    y_true_filtered = y_true[has_labels]
    y_pred_filtered = y_pred[has_labels]
    
    # 过滤掉没有正样本的GO术语
    has_positive = np.sum(y_true_filtered, axis=0) > 0
    if np.sum(has_positive) == 0:
        return 0.0, 0.0
    
    y_true_filtered = y_true_filtered[:, has_positive]
    y_pred_filtered = y_pred_filtered[:, has_positive]
    
    # 计算Fmax：遍历100个阈值
    n = 100
    fmax = 0.0
    best_threshold = 0.0
    prs = []
    rcs = []
    
    for t in range(n + 1):
        threshold = t / n
        pred_bi = (y_pred_filtered > threshold).astype(np.float32)
        
        tp = np.sum(pred_bi * y_true_filtered)
        pred_sum = np.sum(pred_bi)
        true_sum = np.sum(y_true_filtered)
        
        precision = tp / pred_sum if pred_sum > 0 else 0.0
        recall = tp / true_sum if true_sum > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > fmax:
                fmax = f1
                best_threshold = threshold
        
        prs.append(precision)
        rcs.append(recall)
    
    # 计算AUPR
    prs = np.array(prs)
    rcs = np.array(rcs)
    sorted_idx = np.argsort(rcs)
    prs_sorted = prs[sorted_idx]
    rcs_sorted = rcs[sorted_idx]
    
    aupr = np.trapezoid(prs_sorted, rcs_sorted)
    
    return fmax, aupr

# ---------------------------
# Generic npz loader
# ---------------------------
def load_hgat_npz(npz_path: str,
                  name_keys=("name", "names", "protein_names", "protein_ids"),
                  pred_keys=("preds", "predictions", "y_pred"),
                  label_keys=("labels","y_true")):
    npz_path = P.Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    protein_names = None
    for k in name_keys:
        if k in data:
            protein_names = data[k]
            break
    if protein_names is None:
        raise KeyError(f"Cannot find protein name key in {npz_path}. Tried {name_keys}")

    predictions = None
    for k in pred_keys:
        if k in data:
            predictions = data[k]
            break
    if predictions is None:
        raise KeyError(f"Cannot find prediction key in {npz_path}. Tried {pred_keys}")

    labels = None
    for k in label_keys:
        if k in data:
            labels = data[k]
            break 
    if labels is None:
        raise KeyError(f"Cannot find label key in {npz_path}. Tried {label_keys}")

    return protein_names, predictions, labels


# ---------------------------
# ProFun-SOM split-aware loader
# ---------------------------
def load_profunsom_predictions_split(go_name: str,
                                     split: str,
                                     pred_npy_path: str,
                                     dataset_state_pkl: str):
    """
    pred_npy: still provides (2, N, M) for a given ontology/split depending on your export.
    proteins are fetched from dataset_state_dict.pkl split.
    """
    pred_npy_path = P.Path(pred_npy_path)
    if not pred_npy_path.exists():
        raise FileNotFoundError(f"ProFun-SOM pred file not found: {pred_npy_path}")

    pred_data = np.load(pred_npy_path, allow_pickle=True)
    if not (isinstance(pred_data, np.ndarray) and pred_data.shape[0] == 2):
        raise ValueError(f"ProFun-SOM pred format error. Expect (2, N, M), got {pred_data.shape}")

    labels = pred_data[0]
    preds = pred_data[1]

    dataset_state_pkl = P.Path(dataset_state_pkl)
    if not dataset_state_pkl.exists():
        raise FileNotFoundError(f"dataset_state_dict.pkl not found: {dataset_state_pkl}")

    with open(dataset_state_pkl, "rb") as f:
        dataset_state = pickle.load(f)

    ontology_map = {"bp": "biological_process", "cc": "cellular_component", "mf": "molecular_function"}
    ontology = ontology_map[go_name]

    if split not in dataset_state or ontology not in dataset_state[split]:
        raise KeyError(f"Cannot find proteins for split='{split}', ontology='{ontology}' in dataset_state_dict.pkl")

    proteins = np.array(dataset_state[split][ontology]["proteins"])

    # length alignment safeguard
    if len(proteins) != preds.shape[0]:
        min_len = min(len(proteins), preds.shape[0])
        proteins = proteins[:min_len]
        preds = preds[:min_len]
        labels = labels[:min_len]

    return proteins, preds, labels


def load_go_terms_param(go_name: str, namespace_terms_pkl: str, hgat_node_dat_root: str):
    namespace_terms_pkl = P.Path(namespace_terms_pkl)
    if not namespace_terms_pkl.exists():
        raise FileNotFoundError(f"namespace_terms.pkl not found: {namespace_terms_pkl}")

    with open(namespace_terms_pkl, "rb") as f:
        namespace_terms = pickle.load(f)

    ontology_map = {"bp": "biological_process", "cc": "cellular_component", "mf": "molecular_function"}
    ontology = ontology_map[go_name]
    profunsom_go_terms = np.array(namespace_terms[ontology])

    node_path = P.Path(hgat_node_dat_root) / go_name / "node.dat"
    if not node_path.exists():
        raise FileNotFoundError(f"HGAT node.dat not found: {node_path}")

    try:
        node = pd.read_csv(
            node_path, header=None, sep="\t", usecols=[0, 1, 2, 3],
            dtype={3: str}, engine="python", on_bad_lines="skip", quoting=3
        )
    except Exception:
        rows = []
        with open(node_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    rows.append(parts[:4])
        node = pd.DataFrame(rows, columns=["id", "name", "type", "feature"])

    node.columns = ["id", "name", "type", "feature"]
    # robust type parsing
    try:
        t = node["type"].astype(int)
    except Exception:
        t = node["type"].astype(str).astype(int, errors="ignore")
    go_nodes = node[t == 1]
    hgat_go_terms = go_nodes["name"].tolist()

    return profunsom_go_terms, np.array(hgat_go_terms, dtype=object)


def format_tpl(tpl: str, go: str, split: str) -> str:
    return tpl.format(go=go, split=split)


def main():
    parser = argparse.ArgumentParser(
        description="Export predictions in the union label space (split-aware, no fusion)."
    )
    parser.add_argument("--go_name", type=str, default="all", choices=["bp", "cc", "mf", "all"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test", "all"],
                        help="Which split to process.")

    # Use templates so the same script can handle train/valid/test
    parser.add_argument("--zhao_npz_tpl", type=str, required=True,
                        help="Template for Zhao HGAT npz, e.g. /path/HGAT_{go}_{split}.npz or .../{go}/{split}.npz. "
                             "Available fields: {go}, {split}")
    parser.add_argument("--esm2_v3_npz_tpl", type=str, required=True,
                        help="Template for HGAT_ESM2_V3 npz. Available fields: {go}, {split}")

    # ProFun-SOM paths (can also be split-aware via template)
    parser.add_argument("--profunsom_pred_npy_tpl", type=str, required=True,
                        help="Template for ProFun-SOM pred npy (2,N,M). Available fields: {go}, {split}")
    parser.add_argument("--profunsom_dataset_state_pkl", type=str, required=True)
    parser.add_argument("--namespace_terms_pkl", type=str, required=True)
    parser.add_argument("--hgat_node_dat_root", type=str, required=True)

    parser.add_argument("--target_model", type=str, default="esm2_v3",
                        choices=["esm2_v3", "zhao", "profunsom", "all"],
                        help="Which model's predictions to export in union label space.")

    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    go_names = ["bp", "cc", "mf"] if args.go_name == "all" else [args.go_name]
    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    out_root = P.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for go in go_names:
        prof_go_terms, hgat_go_terms = load_go_terms_param(go, args.namespace_terms_pkl, args.hgat_node_dat_root)
        union_go_terms = sorted(list(set(hgat_go_terms) | set(prof_go_terms)))

        for split in splits:
            print("\n" + "=" * 80)
            print(f"[INFO] GO={go.upper()} | split={split}")
            print("=" * 80)

            zhao_npz = format_tpl(args.zhao_npz_tpl, go, split)
            esm2_npz = format_tpl(args.esm2_v3_npz_tpl, go, split)
            prof_npy = format_tpl(args.profunsom_pred_npy_tpl, go, split)

            # Load each source for this split
            zhao_proteins, zhao_preds, zhao_labels = load_hgat_npz(zhao_npz)
            esm2_proteins, esm2_preds, esm2_labels = load_hgat_npz(
                esm2_npz,
                name_keys=("protein_names", "name", "names", "protein_ids"),
                pred_keys=("predictions", "preds", "y_pred"),
                label_keys=("labels", "y_true"),
            )
            prof_proteins, prof_preds, prof_labels = load_profunsom_predictions_split(
                go, split, prof_npy, args.profunsom_dataset_state_pkl
            )

            # Align proteins (split-specific)
            common_proteins, zhao_idx, esm2_idx, prof_idx = align_proteins(
                zhao_proteins, esm2_proteins, prof_proteins
            )

            if len(common_proteins) == 0:
                print("[WARN] common_proteins=0 by name matching; fallback to order-based alignment for THIS split.")
                min_len = min(len(zhao_proteins), len(esm2_proteins), len(prof_proteins))
                common_proteins = [f"{split}_protein_{i}" for i in range(min_len)]
                zhao_idx = list(range(min_len))
                esm2_idx = list(range(min_len))
                prof_idx = list(range(min_len))
                print(f"[INFO] Using first {min_len} proteins (order-aligned).")

            # Slice to common proteins
            zhao_preds_c = zhao_preds[zhao_idx]
            zhao_labels_c = zhao_labels[zhao_idx]

            esm2_preds_c = esm2_preds[esm2_idx]
            esm2_labels_c = esm2_labels[esm2_idx]

            prof_preds_c = prof_preds[prof_idx]
            prof_labels_c = prof_labels[prof_idx]

            # Project to union label space
            zhao_preds_union = align_labels(zhao_preds_c, hgat_go_terms, union_go_terms, fill_value=0.0)
            zhao_labels_union = align_labels(zhao_labels_c, hgat_go_terms, union_go_terms, fill_value=0.0)

            esm2_preds_union = align_labels(esm2_preds_c, hgat_go_terms, union_go_terms, fill_value=0.0)
            esm2_labels_union = align_labels(esm2_labels_c, hgat_go_terms, union_go_terms, fill_value=0.0)

            prof_preds_union = align_labels(prof_preds_c, prof_go_terms, union_go_terms, fill_value=0.0)
            prof_labels_union = align_labels(prof_labels_c, prof_go_terms, union_go_terms, fill_value=0.0)

            union_labels = np.maximum(np.maximum(zhao_labels_union, esm2_labels_union), prof_labels_union)

            if args.target_model == "esm2_v3":
                target_preds_union = esm2_preds_union
            elif args.target_model == "zhao":
                target_preds_union = zhao_preds_union
            elif args.target_model == "profunsom":
                target_preds_union = prof_preds_union
            else:
                # save all
                target_preds_union = {
                    "zhao": zhao_preds_union,
                    "esm2_v3": esm2_preds_union,
                    "profunsom": prof_preds_union,
                }

            # Save to out_dir/go/split/*
            out_dir = out_root / go / split
            out_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(target_preds_union, dict):
                np.savez_compressed(
                    out_dir / "target_preds_union.npz",
                    zhao=target_preds_union["zhao"],
                    esm2_v3=target_preds_union["esm2_v3"],
                    profunsom=target_preds_union["profunsom"],
                )
            else:
                np.save(out_dir / f"target_preds_union_{args.target_model}.npy", target_preds_union)
            np.save(out_dir / "union_go_terms.npy", np.array(union_go_terms, dtype=object))
            np.save(out_dir / "common_proteins.npy", np.array(common_proteins, dtype=object))
            np.save(out_dir / "union_labels.npy", union_labels)

            with open(out_dir / "meta.txt", "w") as f:
                f.write(f"go={go}\n")
                f.write(f"split={split}\n")
                f.write(f"target_model={args.target_model}\n")
                f.write(f"n_proteins={len(common_proteins)}\n")
                f.write(f"n_union_go_terms={len(union_go_terms)}\n")
                f.write(f"zhao_npz={zhao_npz}\n")
                f.write(f"esm2_v3_npz={esm2_npz}\n")
                f.write(f"profunsom_pred_npy={prof_npy}\n")

            print(f"[INFO] Saved: {out_dir}")

if __name__ == "__main__":
    main()
