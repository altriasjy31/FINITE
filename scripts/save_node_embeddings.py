#!/usr/bin/env python3
"""
Save node embeddings (final HGAT hidden states) for each split (train/valid/test).

For each split:
  - Build the inductive subgraph: dataset.train_g / valid_g / test_g
  - Run HGAT_ESM2_V3 full-graph forward (same logic as model.forward, but we return h dict)
  - Build:
      protein_name2emb: {protein_name: np.ndarray(hidden_dim)}
      go_name2emb:      {GO_ID: np.ndarray(hidden_dim)}
  - For proteins: only keep the *seed* proteins belonging to that split
                  (neighbors from other splits are filtered out)
  - For GO terms: keep all GO nodes present in that split subgraph

Output files (per split, per namespace):
  <outdir>/{go_name}_{split}_protein_name2emb.pkl
  <outdir>/{go_name}_{split}_go_name2emb.pkl
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import dgl

# ---------------------------------------------------------------------
# Project imports (assume this script is under <proj_root>/scripts/)
# ---------------------------------------------------------------------
PRJ_ROOT = Path(__file__).resolve().parents[1]
if str(PRJ_ROOT) not in sys.path:
    sys.path.append(str(PRJ_ROOT))

from dataset.inductive_learning_dataset import DBLPDataset
from models.nnHGAT_esm2_v3 import HGAT_ESM2_V3, to_hetero_feat


# ---------------------------------------------------------------------
# Utilities to load ESM2 features and node names
# ---------------------------------------------------------------------
def load_esm2_features(go_name: str, device: torch.device):
    """
    Load global ESM2 features for protein and GO.
    Adjust the path if your data layout is different.
    """
    esm2_dir = PRJ_ROOT / "data" / go_name / "esm2_features"
    protein_path = esm2_dir / "esm2_protein_features.npy"
    go_path = esm2_dir / "esm2_go_features.npy"

    esm2_protein = None
    esm2_go = None

    if protein_path.exists():
        arr = np.load(protein_path)
        esm2_protein = torch.as_tensor(arr, dtype=torch.float32, device=device)
        print(f"[INFO] Loaded ESM2 protein features from {protein_path} shape={esm2_protein.shape}")
    else:
        print(f"[WARN] ESM2 protein features not found: {protein_path}")

    if go_path.exists():
        arr = np.load(go_path)
        esm2_go = torch.as_tensor(arr, dtype=torch.float32, device=device)
        print(f"[INFO] Loaded ESM2 GO features from {go_path} shape={esm2_go.shape}")
    else:
        print(f"[WARN] ESM2 GO features not found: {go_path}")

    return esm2_protein, esm2_go


def load_node_names(go_name: str):
    """
    Reproduce trainer.load_node_names:
      - protein_name_dict: {global_protein_id: protein_name}
      - go_terms_list:     [GO_ID] index = global_go_id
    """
    node_path = PRJ_ROOT / "data" / go_name / "node.dat"
    node = pd.read_csv(node_path, header=None, sep="\t", dtype={3: str})
    node.columns = ["id", "name", "type", "feature"]

    protein_nodes = node[node["type"] == 0].reset_index(drop=True)
    protein_name_dict = {i: n for i, n in enumerate(protein_nodes["name"].tolist())}

    go_nodes = node[node["type"] == 1].reset_index(drop=True)
    go_terms_list = go_nodes["name"].tolist()

    print(f"[INFO] Node names loaded: #protein={len(protein_name_dict)} #go={len(go_terms_list)}")
    return protein_name_dict, go_terms_list


# ---------------------------------------------------------------------
# Core: compute final HGAT embeddings for a given heterograph
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_graph_embeddings(
    model: HGAT_ESM2_V3,
    g: dgl.DGLHeteroGraph,
    h_msa_dict: dict,
    esm2_protein_global: torch.Tensor | None,
    esm2_go_global: torch.Tensor | None,
):
    """
    Re-implements HGAT_ESM2_V3.forward but returns the final h dict
    instead of logits.

    g:   hetero subgraph (train_g/valid_g/test_g), already on device
    h_msa_dict: g.ndata['h'] for each ntype
    esm2_*_global: full-graph ESM2 features (indexed by parent/global id)
    """
    device = next(model.parameters()).device
    g = g.to(device)

    # Build esm2_dict for this subgraph via parent/global ids
    esm2_dict = {}
    if esm2_protein_global is not None and "protein" in g.ntypes:
        p_parent = g.nodes["protein"].data[dgl.NID]  # global protein ids
        esm2_dict["protein"] = esm2_protein_global[p_parent]

    if esm2_go_global is not None and "go_annotation" in g.ntypes:
        g_parent = g.nodes["go_annotation"].data[dgl.NID]  # global go ids
        esm2_dict["go_annotation"] = esm2_go_global[g_parent]

    # 1) Fuse MSA + ESM2 -> hidden_dim
    h = {}
    for ntype in g.ntypes:
        x_msa = h_msa_dict[ntype].to(device)
        x_esm2 = esm2_dict.get(ntype, None)
        h[ntype] = model._fuse(ntype, x_msa, x_esm2)

    # 2) HGAT layers (type + node attention)
    for l in range(model.num_layers):
        alpha_dict = model.type_attn[l](g, h)

        with g.local_scope():
            g.ndata["h"] = h
            g.edata["alpha"] = alpha_dict
            gh = dgl.to_homogeneous(g, ndata=["h"], edata=["alpha"])
            x = gh.ndata["h"]
            x_out = model.node_attn[l](gh, x)
            h = to_hetero_feat(x_out, gh.ndata["_TYPE"], g.ntypes)

    return h  # {"protein": (N_p, hidden), "go_annotation": (N_go, hidden)}


# ---------------------------------------------------------------------
# Build dicts: name -> embedding (per split)
# ---------------------------------------------------------------------
def build_name2emb_dicts_for_split(
    split_name: str,
    dataset: DBLPDataset,
    model: HGAT_ESM2_V3,
    esm2_protein_global: torch.Tensor | None,
    esm2_go_global: torch.Tensor | None,
    protein_name_dict: dict[int, str],
    go_terms_list: list[str],
):
    """
    For one split (train/valid/test), compute embeddings and return:
      protein_name2emb, go_name2emb
    """
    if split_name == "train":
        g_split = dataset.train_g
        split_parent_proteins = dataset.train_idx  # global ids
    elif split_name == "valid":
        g_split = dataset.valid_g
        split_parent_proteins = dataset.valid_idx
    elif split_name == "test":
        g_split = dataset.test_g
        split_parent_proteins = dataset.test_idx
    else:
        raise ValueError(f"Unknown split: {split_name}")

    device = next(model.parameters()).device
    g_split = g_split.to(device)

    # Base features (ProFun-SOM / MSA) are stored in ndata['h']
    h_msa_dict = {ntype: g_split.ndata["h"][ntype] for ntype in g_split.ntypes}

    # Compute final HGAT embeddings
    h_final = compute_graph_embeddings(
        model,
        g_split,
        h_msa_dict,
        esm2_protein_global,
        esm2_go_global,
    )

    # ----------------- Protein dict (only seed proteins of this split) -----------------
    protein_name2emb: dict[str, np.ndarray] = {}
    if "protein" in g_split.ntypes:
        prot_emb = h_final["protein"].detach().cpu().numpy()
        prot_parent_ids = g_split.nodes["protein"].data[dgl.NID].detach().cpu().numpy()
        split_parent_set = set(split_parent_proteins.detach().cpu().tolist())

        for row_idx, parent_id in enumerate(prot_parent_ids):
            if int(parent_id) not in split_parent_set:
                # Neighbor protein from other split -> skip
                continue
            name = protein_name_dict.get(int(parent_id))
            if name is None:
                continue
            protein_name2emb[name] = prot_emb[row_idx]

    # ----------------- GO dict (all GO nodes in this subgraph) -----------------
    go_name2emb: dict[str, np.ndarray] = {}
    if "go_annotation" in g_split.ntypes:
        go_emb = h_final["go_annotation"].detach().cpu().numpy()
        go_parent_ids = g_split.nodes["go_annotation"].data[dgl.NID].detach().cpu().numpy()

        for row_idx, parent_id in enumerate(go_parent_ids):
            idx = int(parent_id)
            if 0 <= idx < len(go_terms_list):
                go_id = go_terms_list[idx]
                go_name2emb[go_id] = go_emb[row_idx]

    print(
        f"[INFO] Split={split_name}: "
        f"protein_name2emb={len(protein_name2emb)} "
        f"go_name2emb={len(go_name2emb)}"
    )
    return protein_name2emb, go_name2emb


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--go_name", type=str, default="bp", choices=["bp", "cc", "mf"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained HGAT_ESM2_V3 checkpoint (e.g., ..._bestFmax.pt)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "valid", "test", "all"],
                        help="Which split to export embeddings for")
    parser.add_argument("--outdir", type=str, default="node_embeddings",
                        help="Output directory for pickled dictionaries")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size used only to construct DBLPDataset (sampler); "
                             "does not affect full-graph embedding here.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Dataset (builds parent graph + inductive subgraphs)
    print(f"[INFO] Building DBLPDataset for go_name={args.go_name}")
    dataset = DBLPDataset(
        dataset_name="data",
        go_name=args.go_name,
        model_name="HGAT",
        batch_size=args.batch_size,
        device=str(device),
    )

    # 2) Load global ESM2 features
    esm2_protein_global, esm2_go_global = load_esm2_features(args.go_name, device)

    # 3) Load node names
    protein_name_dict, go_terms_list = load_node_names(args.go_name)

    # 4) Build model and load checkpoint
    print("[INFO] Building HGAT_ESM2_V3 model")
    model = HGAT_ESM2_V3(
        ntypes=dataset.node_type,
        num_classes=dataset.go_num,
        protein_feat_dim=dataset.feature_dim,
        go_feat_dim=getattr(dataset, "go_feature_dim", None),
        hidden_dim=256,
        num_layers=2,
        use_esm2=True,
        esm2_protein_dim=1280,
        esm2_go_dim=1280,
    ).to(device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"[INFO] Loaded model weights from {ckpt_path}")

    # 5) Which splits to run
    if args.split == "all":
        splits = ["train", "valid", "test"]
    else:
        splits = [args.split]

    # 6) For each split: compute and save dicts
    for split_name in splits:
        print(f"[INFO] Processing split: {split_name}")
        protein_name2emb, go_name2emb = build_name2emb_dicts_for_split(
            split_name=split_name,
            dataset=dataset,
            model=model,
            esm2_protein_global=esm2_protein_global,
            esm2_go_global=esm2_go_global,
            protein_name_dict=protein_name_dict,
            go_terms_list=go_terms_list,
        )

        prot_path = outdir / f"{args.go_name}_{split_name}_protein_name2emb.pkl"
        go_path = outdir / f"{args.go_name}_{split_name}_go_name2emb.pkl"

        with open(prot_path, "wb") as f:
            pickle.dump(protein_name2emb, f)
        with open(go_path, "wb") as f:
            pickle.dump(go_name2emb, f)

        print(f"[SAVE] {prot_path} ({len(protein_name2emb)} proteins)")
        print(f"[SAVE] {go_path} ({len(go_name2emb)} GO terms)")

    print("[DONE] All requested splits processed.")


if __name__ == "__main__":
    main()