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

#%%
ns = ["cc", "mf", "bp"]
ontology_lst = ["cellular_component", "molecular_function", "biological_process"]

#%%
# laad go term
# %%
label_dir = prj_root / "data"
curr_labels = []
for n in ns:
    label_path = label_dir / n / "go_list.txt"
    with open(label_path, "r") as f:
        labels = f.read().splitlines()
    curr_labels.append(labels)

namespace_terms = dict(zip(ontology_lst, curr_labels))

# %%
nspace_ti = {k: {x: i for i, x in enumerate(v)}
             for k, v in namespace_terms.items()}

#%%
# load best weight
# load checkpoint
ckpt_paths = [prj_root / "models" / f"gcn_ppi_{x}"
              for x in ns]
# ckpt_path = ckpt_root / f"hgat_inductive_{x}"
state_lst = [th.load(p) for p in ckpt_paths]

#%%
# get the linear weight as the embedding for GO terms
# emb_lst = [s["linear.weight"].detach().cpu() for s in state_lst]
emb_lst = [s["conv2.weight"].detach().cpu().transpose(0,1)
           for s in state_lst]
# %%
emb_dict = {ont: {k: v 
                     for k, v in zip(curr_labels[i], emb_lst[i])}
               for i, ont in enumerate(ontology_lst)}
# %%
# save the embedding
emb_path = prj_root / "data" / "gcn_ppi_go_emb.pt"
# emb_path = prj_root / "data" / "gcn_ppi_go_linearemb.pt"
# emb_path = prj_root / "data" / "gcn_ppi_go_convemb.pt"
th.save(emb_dict, emb_path)
# %%