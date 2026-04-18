#%%
import typing as T
import pathlib as P
import json
import pickle
import sys
import os
prj_root = P.Path(__file__).absolute().parent.parent.parent
if (p := str(prj_root)) not in sys.path:
    sys.path.append(p)
import collections as clt
import itertools as it
import functools as ft
import operator as opr
#%%
from dataset.hg_dataset import DBLPDataset
# from models.GCN import GCN
from trainer.ppi_gcn_trainer import GCN
import util.metrics as um
#%%
import torch
import torch as th
import numpy as np
import dgl
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F

#%%
ns = ["cc", "mf", "bp"]
ontology_lst = ["cellular_component", "molecular_function", "biological_process"]

#%%
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = None
        self.hook_layer = None
        
    def hook_fn_linear(self, module, input, output):
        """Hook function to capture input to linear layer"""
        self.features = input[0].detach()  # Store the input to linear layer
        
    def hook_fn_conv(self, module, input, output):
        """Hook function to capture input to conv layer"""
        # For GraphConv, input is (graph, features)
        if isinstance(input, tuple) and len(input) >= 2:
            self.features = input[1].detach()  # Store the feature input to conv layer
        else:
            self.features = input[0].detach()
    
    def determine_hook_layer(self, blocks):
        """Determine which layer to hook based on the forward path"""
        if len(blocks) == 1:
            block = blocks[0]
            num_src_nodes = block.number_of_src_nodes()
            num_dst_nodes = block.number_of_dst_nodes()
            
            if num_src_nodes == num_dst_nodes:
                # Will use linear layer
                return self.model.linear, self.hook_fn_linear
            else:
                # Will use conv2 layer
                return self.model.conv2, self.hook_fn_conv
        else:
            # Normal case: will use conv2 layer
            return self.model.conv2, self.hook_fn_conv
        
    def register_hook(self, blocks):
        """Register hook to the appropriate layer based on forward path"""
        self.hook_layer, hook_fn = self.determine_hook_layer(blocks)
        self.hook = self.hook_layer.register_forward_hook(hook_fn)
        
    def remove_hook(self):
        """Remove the registered hook"""
        if self.hook:
            self.hook.remove()
            
    def extract_features(self, blocks):
        """Extract features using the registered hook"""
        self.register_hook(blocks)
        
        # Run forward pass (this will trigger the hook)
        pred = self.model(blocks)
        
        # Get the captured features
        features = self.features
        
        # Clean up
        self.remove_hook()
        
        return features, pred

#%%
def get_ppi(g):
    node_type = ['protein']
    edge_type = ['interacts_1', '_interacts_1']
    
    ppi_src_1, ppi_dst_1 = g.edges(etype='interacts_1')
    ppi_src_2, ppi_dst_2 = g.edges(etype='_interacts_1')
    
    ppi = dgl.heterograph({
        ('protein', 'interacts_1', 'protein'): (th.tensor(ppi_src_1),th.tensor(ppi_dst_1)),
        ('protein', '_interacts_1', 'protein'): (th.tensor(ppi_src_2),th.tensor(ppi_dst_2))
    })
    ppi.ndata['h'] = g.ndata['h']['protein']
    ppi = dgl.to_homogeneous(ppi, ndata=['h'])
    ppi = dgl.add_self_loop(ppi)
    return ppi
#%%
dt_root = P.Path("./data")
def extract_features_by_split(split: str = "test", bs: int = 64):
    feats_lst = []
    testids_lst = []
    for x in ns:
        device = th.device("cuda" if th.cuda.is_available   () else "cpu")
        dataset = DBLPDataset('data', x, 'gcn_ppi', bs, device, result_label='')
        g, _ = dgl.load_graphs('/data0/zhaojianxiang/data/ppi/ppi.dgl')
        g = dgl.add_reverse_edges(g[0]).to(device)
        features = np.load(dataset.data_path.parent.joinpath('features.npz'))['features']
        features = torch.tensor(features).to(device)
        g.ndata['h'] = features
        g = dgl.remove_self_loop(g)  # 移除已有的自环（如果有）
        g = dgl.add_self_loop(g)    # 添加自环

        model = GCN(num_classes=dataset.go_num, in_feats=dataset.feature_dim, h_feats=64)

        # load checkpoint
        ckpt_root = prj_root / "models"
        ckpt_path = ckpt_root / f"gcn_ppi_{x}"
        # ckpt_path = ckpt_root / f"hgat_inductive_{x}"
        state_dict = th.load(ckpt_path)
        model.load_state_dict(state_dict)

        model = model.to(device)

        model.eval()

        # extract feature
        extractor = FeatureExtractor(model)

        # set loader
        category = dataset.category
        sampler = dgl.dataloading.NeighborSampler([5,5])
        # Select loader based on split
        if split == "train":
            loader = dgl.dataloading.DataLoader(
                    g, dataset.train_idx.to(device), sampler,
                    batch_size=bs, device=device, shuffle=True, drop_last=True)
        elif split == "valid":
            loader = dgl.dataloading.DataLoader(
                    g, dataset.valid_idx.to(device), sampler,
                    batch_size=bs, device=device, shuffle=True, drop_last=True)
        elif split == "test":
            loader = dgl.dataloading.DataLoader(
                        g, dataset.test_idx.to(device), sampler,
                        batch_size=bs, device=device, shuffle=False, drop_last=False)
        else:
            raise ValueError(f"Unknown split: {split}")

        loader_tqdm = tqdm(loader, ncols=120)

        y_trues = []
        y_predicts = []
        prot_indices = []
        features = []
        with torch.no_grad():
            for i, (sample_nodes, seed, blocks) in enumerate(loader_tqdm):
                label = dataset.get_label(seed.cpu().numpy().tolist())
                h_dict = blocks[0].srcdata['h']
                h_dict = h_dict.float()
                feat, pred = extractor.extract_features(blocks)

                label = torch.tensor(label).to(device)

                seed_indices = [torch.where(sample_nodes == x)[0].item() for x in seed]
                feat = feat[seed_indices]

                pred = F.sigmoid(pred)
                y_trues.append(label.to(torch.float16).detach().cpu())
                y_predicts.append(pred.to(torch.float16).detach().cpu())
                features.append(feat.to(torch.float32).detach().cpu())
                prot_indices += seed.cpu().numpy().tolist()
            y_trues = torch.cat(y_trues, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)

        # evaluate
        fmax, aupr = um.fmax_score(y_trues, y_predicts, auprc=True)
        print('fmax:{}, aupr:{}'.format(fmax, aupr))

        feats_lst.append(torch.cat(features, dim=0))
        testids_lst.append(prot_indices)
    
        print(f"Extracted features for {x} in {split} split: {feats_lst[-1].shape}, "
              f"Number of proteins: {len(testids_lst[-1])}")
    # get protein name id
    name_path = prj_root / "data" / "protein_name.txt"
    with open(name_path, "r") as f:
        prot_lst = f.read().splitlines()

    # build a dict for protein name to features'
    featdict_lst = [{prot_lst[j]: feat for j, feat in zip(testids_lst[i], feats)} for i, feats in enumerate (feats_lst)]
    return featdict_lst

#%%
# get feature for each split
split_type = ["train", "valid", "test"]
feat_states = [extract_features_by_split(x)
                for x in split_type]

# %%
# saving features
saving_dir = prj_root / "data" / "derived_feature"

for i,t in enumerate(split_type):
    saving_path = saving_dir / f"gcn_ppi_features_{t}.pt"
    th.save(feat_states[i], saving_path)
# %%
