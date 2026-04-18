# %%
import os
import sys
import typing as T
import itertools as it
import functools as ft
import operator as opr
import collections as clt
import pathlib as P
import json
import pickle
from collections import defaultdict
import bisect

# %%
prj_root = P.Path(__file__).absolute().parent.parent.parent
if (p := str(prj_root)) not in sys.path:
    sys.path.append(p)

# %%
# import helper_functions.helper as H
import util.obo_parser as gp
import util.build_term_mapping as btm

# %%
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import sklearn.manifold as mf
import tqdm
import torch as th
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# %%
ns = ["cc", "mf", "bp"]

# %% [markdown]
# ## load data

# %%
root_path = P.Path("/data0/shaojiangyi/pprogo-flg-1/data/data-netgo")

# %%
# ontpath = somdata_dir.joinpath("ontology_object.pkl")
# with open(ontpath, "rb") as h:
#   ontobj = pickle.load(h)
obopath = root_path.joinpath("go.obo")
ontobj = gp.Ontology(obopath, with_rels=True)

# %% [markdown]
# ### calculate term counts

# %%
lable_path = [f"/data0/shaojiangyi/pprogo-flg-1/lmj_results/results/fused_three_models_esm2_v3_{x}_optimized/union_go_terms.npy" for x in ns]
curr_labels = [np.load(p).tolist() for p in lable_path]
curr_index = {k: {t: j for j, t in enumerate(curr_labels[i])} for i, k in enumerate(ns)}

root_path = P.Path("/data0/zhaojianxiang/data/")
prev_labels = []
for n in ns:
    label_path = root_path / n / "go_list.txt"
    with open(label_path, "r") as f:
        labels = f.read().splitlines()
    prev_labels.append(labels)

root_path = P.Path("/data0/zhaojianxiang/data/")
prot_labels = []
for n in ns:
    y_label = root_path / n / "label.pkl"
    with open(y_label, "rb") as f:
        labels = pickle.load(f)
    tmp = defaultdict(list)
    for k, v in zip(labels["protein"], labels["go"]):
        tmp[k].append(v)
    prot_labels.append(list(tmp.values()))

#%%
# each labels is a dictionary with proteins and go
# proteins contains all the names of proteins
# go contains the GO terms associated with each protein

# using collection counter
term_counts = defaultdict(dict)
for i, (k, vs) in enumerate(zip(ns, prot_labels)):
    go_labels = [[prev_labels[i][x] for x in xs] for xs in vs]
    tmp_counts = clt.Counter(it.chain.from_iterable(go_labels))
    term_counts[k] = {t: tmp_counts.get(t, 0) for t in curr_labels[i]}

# # change lable index to go term
# term_counts = {k: {curr_labels[i][v]: c for v, c in term_counts[k].items()} 
#                for i, k in enumerate(ns)}

# %%
# claulcate term ic
term_ic = defaultdict(dict)
for i, (k, vs) in enumerate(zip(ns, prot_labels)):
    # ontobj.calculate_ic(go_labels)
    print(len(vs))
    term_ic[k] = {term: -np.log2(term_counts[k][term] / len(vs)) if term_counts[k][term] > 0 else 0.0
                  for term in curr_labels[i]}
    

#%%
# convert to dataframe
# term_counts = pd.concat([pd.DataFrame(dict(zip(["gos", "counts"], list(zip(*v.items()))))) for v in term_counts.values()])
term_counts_ic = pd.DataFrame(dict(zip(["gos", "counts", "ic"],
                                       [list(it.chain.from_iterable(curr_labels)),
                                        [term_counts[k][t] for i, k in enumerate(ns) \
                                          for t in curr_labels[i]],
                                        [term_ic[k][t] for i, k in enumerate(ns) \
                                          for t in curr_labels[i]]])))


# %%
term_lst = term_counts_ic.gos.to_list()

# %% [markdown]
# ## ontolgoy depth

# %%
term_positions, term_lgst, term_maxdep = btm.build_ontology_position(ontobj, term_lst=term_lst,
                                                                need_longest_path=True,
                                                                need_maxdep=True)

# %% [markdown]
# ### count depth category

# %%
count = clt.Counter(term_positions.values())

# %%
count

# %%
term_counts_ic = term_counts_ic.assign(position=term_counts_ic.gos.map(term_positions),
                                 maxdep=term_counts_ic.gos.map(term_maxdep),
                                 lgst=term_counts_ic.gos.map(term_lgst))

# %%
term_counts_ic

# %%
term_set = set(term_counts_ic.gos.to_list())

# %% [markdown]
# > saving

# %%
root_path = P.Path("/data0/shaojiangyi/pprogo-flg-1/data/")
fname = "term_counts_with_position-111"
term_path = root_path.joinpath(f"{fname}.pkl")
term_counts_ic.to_pickle(term_path)
# save with json
term_path = root_path.joinpath(f"{fname}.json")
term_counts_ic.to_json(term_path, orient="records", lines=True)

# %%