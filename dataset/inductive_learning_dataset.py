import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import random
import pathlib as P
from collections import Counter, defaultdict
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# -----------------------------
# ID mapping utils
# -----------------------------
def parent_to_subgraph_nids(hg: dgl.DGLHeteroGraph, ntype: str, parent_ids: torch.Tensor) -> torch.Tensor:
    """
    Map parent graph node IDs -> subgraph local node IDs for a given ntype.
    Requires hg.nodes[ntype].data[dgl.NID] (parent IDs aligned with local order).
    """
    if parent_ids.numel() == 0:
        return parent_ids

    # hg_parent_ids = hg.nodes[ntype].data[dgl.NID].detach().cpu()
    hg_parent_ids = hg.nodes[ntype].data["parent_nid"].detach().cpu()
    parent_ids_cpu = parent_ids.detach().cpu()

    mapping = {int(pid): i for i, pid in enumerate(hg_parent_ids.tolist())}

    local = []
    missing = []
    for pid in parent_ids_cpu.tolist():
        pid = int(pid)
        if pid in mapping:
            local.append(mapping[pid])
        else:
            missing.append(pid)

    if missing:
        raise ValueError(
            f"[parent_to_subgraph_nids] {len(missing)} parent ids are NOT in subgraph ntype={ntype}. "
            f"Example missing: {missing[:20]}. "
            f"Likely your split seeds are out of closure-subgraph. Increase hop/fanout or check split files."
        )

    return torch.tensor(local, device=parent_ids.device, dtype=torch.long)


def k_hop_closure_hetero(full_g: dgl.DGLHeteroGraph,
                         seed_nodes_parent: Dict[str, torch.Tensor],
                         k: int = 2,
                         fanout: int = -1) -> Dict[str, torch.Tensor]:
    """
    HGAT-inductive style: collect related nodes via k-hop expansion on hetero graph.
    - full_g: parent graph
    - seed_nodes_parent: {ntype: parent_ids}
    - k: hop
    - fanout: -1 means take all neighbors; else sample up to fanout per hop (faster but approximate)

    Returns:
      dict {ntype: parent_ids_of_related_nodes}
    """
    # ensure unique
    all_nodes: Dict[str, torch.Tensor] = {}
    frontier: Dict[str, torch.Tensor] = {}
    for ntype, ids in seed_nodes_parent.items():
        if ids.numel() == 0:
            continue
        u = torch.unique(ids)
        all_nodes[ntype] = u
        frontier[ntype] = u

    for _ in range(k):
        if len(frontier) == 0:
            break

        # sample neighbors from PARENT graph, input IDs are parent IDs
        sg = dgl.sampling.sample_neighbors(full_g, frontier, fanout=fanout, replace=False)

        # IMPORTANT: sample_neighbors does NOT guarantee dgl.NID exists.
        # compact_graphs will relabel nodes and record parent node IDs in dgl.NID ("_ID").
        sg = dgl.compact_graphs(sg)

        new_frontier: Dict[str, torch.Tensor] = {}
        for ntype in full_g.ntypes:
            if sg.num_nodes(ntype) == 0:
                continue

            pnids = sg.nodes[ntype].data[dgl.NID]
            pnids = torch.unique(pnids)

            if ntype in all_nodes:
                prev = all_nodes[ntype]
                # find newly added nodes
                mask = ~torch.isin(pnids, prev)
                added = pnids[mask]
                merged = torch.unique(torch.cat([prev, pnids]))
            else:
                added = pnids
                merged = pnids

            all_nodes[ntype] = merged
            if added.numel() > 0:
                new_frontier[ntype] = added

        frontier = new_frontier

    return all_nodes


def build_split_subgraph(full_g: dgl.DGLHeteroGraph,
                         protein_seeds_parent: torch.Tensor,
                         hop: int = 2,
                         fanout: int = -1,
                         include_all_go: bool = False) -> dgl.DGLHeteroGraph:
    """
    Build inductive subgraph around a set of protein seeds using k-hop hetero closure.
    Optionally include all GO nodes (not recommended for strict closure; use only if you want full GO space always present).
    """
    related = k_hop_closure_hetero(full_g, {'protein': protein_seeds_parent}, k=hop, fanout=fanout)

    if include_all_go:
        related['go_annotation'] = torch.arange(full_g.num_nodes('go_annotation'),
                                               device=protein_seeds_parent.device)

    # node_subgraph will relabel nodes and attach parent IDs into dgl.NID
    sub_g = dgl.node_subgraph(full_g, related, relabel_nodes=True)

    # 固化 full graph 原始ID
    for ntype in sub_g.ntypes:
        if dgl.NID in sub_g.nodes[ntype].data:
            sub_g.nodes[ntype].data["parent_nid"] = sub_g.nodes[ntype].data[dgl.NID]
        elif "parent_nid" in sub_g.nodes[ntype].data:
            pass
        else:
            raise RuntimeError(f"Missing parent id field for ntype={ntype}")

    return sub_g


# -----------------------------
# Dataset
# -----------------------------
class DBLPDataset(Dataset):
    """
    Inductive HGAT dataset (protein-go) with HGAT-repo style inductive closure subgraphs.

    Key points vs your old version:
      1) Train/Valid/Test each gets its own closure subgraph (no val leakage into train graph)
      2) Train labels built from link.dat ONLY (no link.dat.test leakage)
      3) Seeds are mapped parent->local using dgl.NID
      4) test_g is built the same way as train_g/valid_g (no out_subgraph-only protein expansion)
    """

    def __init__(self, dataset_name: str, go_name: str, model_name: str,
                 batch_size: int, device: str,
                 hop: int = 2,
                 closure_fanout: int = -1,
                 include_all_go: bool = False,
                 sampler_type: str = "hgat",
                 dataset_fanouts: Optional[List[int]] = None):

        super().__init__()
        self.go_name = go_name
        self.device = device
        self.batch_size = batch_size

        self.sampler_type = sampler_type.lower()
        self.dataset_fanouts = dataset_fanouts

        self.data_path = P.Path(__file__).absolute().parent.parent.joinpath(f'{dataset_name}/{go_name}')
        self.result_path = P.Path(__file__).absolute().parent.parent.joinpath(f'my_results/inductive_{model_name}_{go_name}')
        self.data_output_path = P.Path(f'./{dataset_name}/{go_name}')

        # 1) Build parent graph ONCE
        self.g = self.create_graph_dynamic().to(self.device)

        # 2) Read splits (parent/global protein ids)
        self.train_idx, self.valid_idx, self.test_idx = self.get_split_from_lists()

        # 3) Build TRAIN-ONLY label dict (IMPORTANT: no test leakage)
        self.label_dict_train = self.load_label_dict_train_only(self.data_path.joinpath('label_train_only.pkl'))

        # 3b) Build EVAL label dict (train + test edges) for metrics only
        self.label_dict_eval = self.load_label_dict_eval(self.data_path.joinpath('label_eval.pkl'))

        # 4) Build inductive closure subgraphs for each split
        self.train_g = build_split_subgraph(self.g, self.train_idx.to(self.device),
                                            hop=hop, fanout=closure_fanout, include_all_go=include_all_go)
        self.valid_g = build_split_subgraph(self.g, self.valid_idx.to(self.device),
                                            hop=hop, fanout=closure_fanout, include_all_go=include_all_go)
        self.test_g = build_split_subgraph(self.g, self.test_idx.to(self.device),
                                           hop=hop, fanout=closure_fanout, include_all_go=include_all_go)

        # 5) seeds: parent -> local
        self.train_seeds = parent_to_subgraph_nids(self.train_g, 'protein', self.train_idx.to(self.device))
        self.valid_seeds = parent_to_subgraph_nids(self.valid_g, 'protein', self.valid_idx.to(self.device))
        self.test_seeds = parent_to_subgraph_nids(self.test_g, 'protein', self.test_idx.to(self.device))

        # 6) meta
        self.node_type = ['protein', 'go_annotation']
        self.category = 'protein'

        self.feature_dim = int(self.g.ndata['h']['protein'].shape[1]) if self.g.num_nodes('protein') > 0 else 0
        self.protein_num = self.g.num_nodes('protein')
        self.go_num = self.g.num_nodes('go_annotation')
        self.go_feature_dim = int(self.g.ndata['h']['go_annotation'].shape[1]) if self.g.num_nodes('go_annotation') > 0 else self.go_num

        # 7) DGL Sampler/Loaders
        # sampler_type:
        #   "hgat"      -> 兼容当前 HGAT 训练（单层 fanout=[5]）
        #   "graphsage" -> GraphSAGE 风格，多层 fanout, 默认 [10, 10]
        fanouts = self.dataset_fanouts if self.dataset_fanouts is not None else [10,10]
        if self.sampler_type == "graphsage":
            # GraphSAGE 通常用多层采样
            self.sampler = dgl.dataloading.NeighborSampler(fanouts)
            print(f"[INFO] Using GraphSAGE sampler with fanouts={fanouts}")
        else:
            # 默认：保留原 sampler 行为
            self.sampler = dgl.dataloading.NeighborSampler(fanouts[:1]) # single layer
            print(f"[INFO] Using HGAT sampler NeighborSampler({fanouts[:1]})")

        self.train_loader = dgl.dataloading.DataLoader(
            self.train_g,
            {'protein': self.train_seeds},
            self.sampler,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=True
        )
        self.valid_loader = dgl.dataloading.DataLoader(
            self.valid_g,
            {'protein': self.valid_seeds},
            self.sampler,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=False
        )
        self.test_loader = dgl.dataloading.DataLoader(
            self.test_g,
            {'protein': self.test_seeds},
            self.sampler,
            batch_size=self.batch_size,
            device=self.device,
            shuffle=False
        )

        # Debug prints (optional)
        print(f"[INFO] Parent graph: #protein={self.protein_num}, #go={self.go_num}")
        print(f"[INFO] Split sizes: train={len(self.train_idx)} valid={len(self.valid_idx)} test={len(self.test_idx)}")
        print(f"[INFO] Subgraphs (nodes): "
              f"train_p={self.train_g.num_nodes('protein')} train_go={self.train_g.num_nodes('go_annotation')}; "
              f"valid_p={self.valid_g.num_nodes('protein')} valid_go={self.valid_g.num_nodes('go_annotation')}; "
              f"test_p={self.test_g.num_nodes('protein')} test_go={self.test_g.num_nodes('go_annotation')}")

    # -----------------------------
    # split
    # -----------------------------
    def get_split_from_lists(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_path = self.data_path.joinpath("node.dat")
        node = pd.read_csv(node_path, header=None, sep="\t", dtype={3: str})

        proteins = node[node[2] == 0][1].tolist()
        protein_to_idx = {p: i for i, p in enumerate(proteins)}

        train_txt = self.data_path.joinpath("train_protein_ids.txt")
        valid_txt = self.data_path.joinpath("valid_protein_ids.txt")
        test_txt  = self.data_path.joinpath("test_protein_ids.txt")

        def _read_name_list(txt_path: P.Path) -> List[str]:
            if not txt_path.exists():
                raise FileNotFoundError(f"Missing split file: {txt_path}")
            names = []
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        names.append(s)
            return names

        train_names = _read_name_list(train_txt)
        valid_names = _read_name_list(valid_txt)
        test_names  = _read_name_list(test_txt)

        def map_names(names: List[str], split: str) -> List[int]:
            idx = []
            missing = []
            seen = set()
            for p in names:
                if p in seen:
                    continue
                seen.add(p)
                if p in protein_to_idx:
                    idx.append(protein_to_idx[p])
                else:
                    missing.append(p)
            if missing:
                print(f"[WARN] {split}: {len(missing)}/{len(names)} proteins not found in node.dat. Example: {missing[:10]}")
            return idx

        train_idx = map_names(train_names, "train")
        valid_idx = map_names(valid_names, "valid")
        test_idx  = map_names(test_names,  "test")

        # basic sanity
        inter = set(train_idx) & set(valid_idx) | set(train_idx) & set(test_idx) | set(valid_idx) & set(test_idx)
        if len(inter) > 0:
            print(f"[WARN] split overlap exists in id space. Example overlap: {list(inter)[:10]}")

        train_t = torch.tensor(train_idx, dtype=torch.long, device=self.device)
        valid_t = torch.tensor(valid_idx, dtype=torch.long, device=self.device)
        test_t  = torch.tensor(test_idx,  dtype=torch.long, device=self.device)

        return train_t, valid_t, test_t

    # -----------------------------
    # labels (train-only, no leakage)
    # -----------------------------
    def load_label_dict_train_only(self, cache_path: P.Path) -> Dict[int, List[int]]:
        """
        Build protein->go label map from link.dat ONLY.
        This avoids leaking test labels into training/validation loss.
        """
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t', dtype={3: str})
        link = pd.read_csv(self.data_path.joinpath('link.dat'), header=None, sep='\t')

        # only p-g edges (type==0) as labels
        link = link[link[2] == 0].copy()

        protein_num = node[node[2] == 0].shape[0]
        # go ids in link.dat are global node ids (protein offset) -> convert to go local id
        link[1] = link[1] - protein_num

        protein_ids = link[0].astype(int).tolist()
        go_ids = link[1].astype(int).tolist()

        protein_to_go: Dict[int, List[int]] = {}
        for p, g in zip(protein_ids, go_ids):
            protein_to_go.setdefault(int(p), []).append(int(g))

        with open(cache_path, 'wb') as f:
            pickle.dump(protein_to_go, f)

        return protein_to_go

    def load_label_dict_eval(self, cache_path: P.Path) -> Dict[int, List[int]]:
        """
        Build protein->go label map from BOTH link.dat and link.dat.test
        用于评估（valid/test metrics），不会参与训练反向传播。
        """
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t', dtype={3: str})
        protein_num = node[node[2] == 0].shape[0]

        # 训练边
        link_train = pd.read_csv(self.data_path.joinpath('link.dat'), header=None, sep='\t')
        link_train = link_train[link_train[2] == 0].copy()

        # 测试边（如果存在）
        link_test_path = self.data_path.joinpath('link.dat.test')
        if link_test_path.exists():
            link_test = pd.read_csv(link_test_path, header=None, sep='\t')
            link_test = link_test[link_test[2] == 0].copy()
            link_all = pd.concat([link_train, link_test], axis=0, ignore_index=True)
        else:
            link_all = link_train

        # go ids: global node ids -> go local id（减掉 protein 个数）
        link_all[1] = link_all[1] - protein_num

        protein_ids = link_all[0].astype(int).tolist()
        go_ids = link_all[1].astype(int).tolist()

        protein_to_go: Dict[int, List[int]] = {}
        for p, g in zip(protein_ids, go_ids):
            protein_to_go.setdefault(int(p), []).append(int(g))

        with open(cache_path, 'wb') as f:
            pickle.dump(protein_to_go, f)

        return protein_to_go

    def get_label(self, protein_parent_ids: List[int]) -> List[List[int]]:
        """
        Return multi-hot labels for given protein IDs in PARENT graph ID space.
        Uses TRAIN-ONLY label dict to avoid leakage.
        """
        label_list = [[0] * self.go_num for _ in range(len(protein_parent_ids))]
        protein_idx_map = {pid: i for i, pid in enumerate(protein_parent_ids)}

        for pid in protein_parent_ids:
            if pid in self.label_dict_train:
                for go in self.label_dict_train[pid]:
                    if 0 <= go < self.go_num:
                        label_list[protein_idx_map[pid]][go] = 1
        return label_list

    def get_label_eval(self, protein_parent_ids: List[int]) -> List[List[int]]:
        """
        Eval 版本的标签获取：使用 train+test 的 label_dict_eval。
        只在验证/测试评估和独立预测脚本中使用。
        """
        label_list = [[0] * self.go_num for _ in range(len(protein_parent_ids))]
        protein_idx_map = {pid: i for i, pid in enumerate(protein_parent_ids)}

        for pid in protein_parent_ids:
            if pid in self.label_dict_eval:
                for go in self.label_dict_eval[pid]:
                    if 0 <= go < self.go_num:
                        label_list[protein_idx_map[pid]][go] = 1
        return label_list

    # -----------------------------
    # graph construction (keep your original dynamic builder)
    # -----------------------------
    def create_graph_dynamic(self, esm_name='esm1', top=100) -> dgl.DGLHeteroGraph:
        """
        Keep your existing logic but ensure:
          - node features align with node ids
          - go ids/protein ids are in correct local space
          - dgl.NID set to parent IDs (0..N-1)
        """
        data_path = self.data_path
        node = pd.read_csv(data_path.joinpath('node.dat'), header=None, sep='\t', dtype={3: str})
        node.columns = ['id', 'name', 'type', 'feature']
        node['id'] = node.groupby('type').cumcount()

        # ---- protein features ----
        first_protein_feature = None
        for _, row in node.iterrows():
            if row['type'] == 0 and pd.notna(row['feature']) and str(row['feature']).strip():
                first_protein_feature = str(row['feature']).split(',')
                break
        if first_protein_feature is None:
            raise ValueError("No protein feature found in node.dat")

        feature_dim = len(first_protein_feature)
        protein_features = []
        go_raw_features = []

        for _, row in node.iterrows():
            if row['type'] == 0:
                if pd.notna(row['feature']) and str(row['feature']).strip():
                    protein_features.append(list(map(float, str(row['feature']).split(','))))
                else:
                    protein_features.append([0.0] * feature_dim)
            else:
                go_raw_features.append(row['feature'])

        protein_tensor = torch.tensor(protein_features, dtype=torch.float32)
        protein_num_from_node = protein_tensor.shape[0]

        # ---- read links to determine required node counts ----
        link_0 = pd.read_csv(data_path.joinpath('esm1_mmseqs_pg_100.dat'), header=None, sep='\t')
        link_1 = pd.read_csv(data_path.parent.joinpath('esm1_mmseqs_pp_100.dat'), header=None, sep='\t')

        max_protein_id = int(max(link_0[0].max(), link_1[0].max(), link_1[1].max()))
        required_protein_num = max_protein_id + 1
        if protein_num_from_node < required_protein_num:
            pad = torch.zeros((required_protein_num - protein_num_from_node, protein_tensor.shape[1]), dtype=torch.float32)
            protein_tensor = torch.cat([protein_tensor, pad], dim=0)

        # ---- GO features ----
        link_2 = pd.read_pickle(data_path.joinpath('go_rel.pkl'))
        link_2.columns = [0, 1, 2, 3]
        max_go_id = int(max(link_0[1].max(), link_2[0].max(), link_2[1].max()))
        required_go_num = max_go_id + 1

        go_features_list = []
        for gf in go_raw_features:
            if pd.notna(gf) and str(gf).strip():
                go_features_list.append(list(map(float, str(gf).split(','))))
            else:
                go_features_list.append(None)

        if len(go_features_list) > 0 and all(f is not None for f in go_features_list):
            go_tensor = torch.tensor(go_features_list, dtype=torch.float32)
            if go_tensor.shape[0] < required_go_num:
                pad = torch.zeros((required_go_num - go_tensor.shape[0], go_tensor.shape[1]), dtype=torch.float32)
                go_tensor = torch.cat([go_tensor, pad], dim=0)
        else:
            # fallback one-hot (very large, but keep backward compatibility)
            go_indices = torch.arange(required_go_num, dtype=torch.long)
            go_tensor = F.one_hot(go_indices, num_classes=required_go_num).to(torch.float32)

        feature_dict = {'protein': protein_tensor, 'go_annotation': go_tensor}

        # ---- edges ----
        p2g_src = link_0[0].astype(int).values
        p2g_dst = link_0[1].astype(int).values  # already go-local ids in your data
        p2p_src = link_1[0].astype(int).values
        p2p_dst = link_1[1].astype(int).values
        g2g_src = link_2[0].astype(int).values
        g2g_dst = link_2[1].astype(int).values

        graph_data = {
            ('protein', 'interacts_0', 'go_annotation'): (torch.tensor(p2g_src), torch.tensor(p2g_dst)),
            ('go_annotation', '_interacts_0', 'protein'): (torch.tensor(p2g_dst), torch.tensor(p2g_src)),
            ('protein', 'interacts_1', 'protein'): (torch.tensor(p2p_src), torch.tensor(p2p_dst)),
            ('protein', '_interacts_1', 'protein'): (torch.tensor(p2p_dst), torch.tensor(p2p_src)),
            ('go_annotation', 'interacts_2', 'go_annotation'): (torch.tensor(g2g_src), torch.tensor(g2g_dst)),
            ('go_annotation', '_interacts_2', 'go_annotation'): (torch.tensor(g2g_dst), torch.tensor(g2g_src)),
        }

        g = dgl.heterograph(graph_data)
        g.ndata['h'] = feature_dict

        # parent IDs are 0..N-1 within this graph
        for ntype in g.ntypes:
            g.nodes[ntype].data["parent_nid"] = torch.arange(g.num_nodes(ntype), dtype=torch.long)


        return g
