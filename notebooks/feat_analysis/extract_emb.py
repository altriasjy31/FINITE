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
from models.HGAT import HGAT
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
        
    def hook_fn(self, module, input, output):
        """Hook function to capture input to FC layer"""
        self.features = input[0].detach()  # Store the input to FC layer
        
    def register_hook(self):
        """Register hook to the final FC layer"""
        assert hasattr(self.model, "output"), "Model does not have 'output' attribute"

        self.hook =  getattr(self.model, "output").register_forward_hook(self.hook_fn)
        
    def remove_hook(self):
        """Remove the registered hook"""
        if self.hook:
            self.hook.remove()
            
    def extract_features(self, hg, h_dict):
        """Extract features using the registered hook"""
        self.register_hook()
        
        # Run forward pass (this will trigger the hook)
        pred = self.model(hg, h_dict)
        
        # Get the captured features
        features = self.features
        
        # Clean up
        self.remove_hook()
        
        return features, pred

# Usage example:
# extractor = FeatureExtractor(model)
# features = extractor.extract_features(hg, h_dict)

#%%
dt_root = P.Path("./data")
def extract_features_by_split(split: str = "test", bs: int = 64):
    """
    Extract features for a given data split ('train', 'valid', or 'test').

    Args:
        split (str): Which split to extract features from. One of 'train', 'valid', 'test'.
        bs (int): Batch size.

    Returns:
        List[Dict[str, torch.Tensor]]: List of dicts mapping protein names to feature tensors for each ontology.
    """
    feats_lst = []
    testids_lst = []
    for x in ns:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        dataset = DBLPDataset(dt_root, x, 'HGAT', bs, device)

        g = dataset.g.to(device)

        model = HGAT(dataset.node_type, num_classes=dataset.go_num, feature_dim=dataset.feature_dim, hidden_dim=128, num_layers=2)

        # load checkpoint
        ckpt_root = prj_root / "models"
        ckpt_path = ckpt_root / f"hgat_{x}_best"
        state_dict = th.load(ckpt_path)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        extractor = FeatureExtractor(model)

        # Select loader based on split
        if split == "train":
            loader = dataset.train_loader
        elif split == "valid":
            loader = dataset.valid_loader
        elif split == "test":
            loader = dataset.test_loader
        else:
            raise ValueError(f"Unknown split: {split}")

        loader_tqdm = tqdm(loader, ncols=120)

        y_trues = []
        y_predicts = []
        prot_indices = []
        features = []
        with torch.no_grad():
            for i, (sample_nodes, seed, blocks) in enumerate(loader_tqdm):
                with autocast():
                    seed = seed['protein']
                    subgraph = dgl.node_subgraph(g, sample_nodes)
                    label = dataset.get_label(seed.cpu().numpy().tolist())
                    h_dict = subgraph.ndata['h']
                    feat, pred = extractor.extract_features(subgraph, h_dict)

                    seed_indices = [torch.where(sample_nodes['protein'] == x)[0].item() for x in seed]

                    feat = feat[seed_indices]
                    pred = pred['protein'][seed_indices]
                    label = torch.tensor(label).to(device)
                    prot_indices += seed.cpu().numpy().tolist()

                    if torch.isnan(pred).any():
                        print(pred)
                        raise ValueError("nan")

                    pred = F.sigmoid(pred)
                    y_trues.append(label.to(torch.float32).detach().cpu())
                    y_predicts.append(pred.to(torch.float32).detach().cpu())
                    features.append(feat.to(torch.float32).detach().cpu())
            y_trues = torch.cat(y_trues, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)

        # evaluate
        fmax, aupr = um.fmax_score(y_trues, y_predicts, auprc=True)
        print(f'Ontology: {x} | fmax: {fmax}, aupr: {aupr}')

        feats_lst.append(torch.cat(features, dim=0))
        testids_lst.append(prot_indices)

    # get protein name id
    name_path = prj_root / "data" / "protein_name.txt"
    with open(name_path, "r") as f:
        prot_lst = f.read().splitlines()

    # build a dict for protein name to features'
    featdict_lst = [{prot_lst[j]: feat for j, feat in zip(testids_lst[i], feats)} for i, feats in enumerate(feats_lst)]
    return featdict_lst

#%%
# get feature for each split
split_type = ["train", "valid", "test"]
feat_states = [extract_features_by_split(x)
                for x in split_type]

# %%
# saving features
saving_dir = prj_root / "data" / "derived_feature"
# feat_state = dict(zip(ontology_lst, featdict_lst))
# saving_path = saving_dir / "hgat_features.pt"
for i,t in enumerate(split_type):
    saving_path = saving_dir / f"hgat_features_{t}.pt"
    th.save(feat_states[i], saving_path)
# %%
