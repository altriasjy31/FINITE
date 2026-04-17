import os
import sys
import torch as th
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
from .hg_sampler import *
from pathlib import Path
from models import HeteroLinear

# define the dataset for link prediction

class DBLPDataset(Dataset):
    def __init__(self, dataset_name, go_name, model_name, batch_size, device, result_label='', sampler='neighbor_sampler'):
        """_summary_

        Args:
            dataset_name (_type_): _description_
            go_name (_type_): _description_
            model_name (_type_): _description_
            batch_size (_type_): _description_
            device (_type_): _description_
            sampler (str, optional): 
                sampler:'neighbor_sampler', 'han_sampler', ''. Defaults to 'neighbor_sampler'.
        """
        self.data_path = P.Path(__file__).absolute().parent.parent.joinpath('{}/{}'.format(dataset_name, go_name))
        self.result_path = P.Path(__file__).absolute().parent.parent.joinpath('results/{}_{}_{}'.format(model_name, go_name, result_label))
        self.data_output_path = P.Path('./{}/{}'.format(dataset_name, go_name))
        self.g_path = self.data_path.joinpath('graph.dgl')
        # self.g = self.create_graph().to(device)
        self.g = self.create_graph_dynamic().to(device)
        self.feature_dim = len(self.g.ndata['h']['protein'][0])
        self.protein_num = self.g.number_of_nodes('protein')
        self.go_num = self.g.number_of_nodes('go_annotation')
        self.edge_num = 0
        for etype in self.g.etypes:
            self.edge_num += self.g.number_of_edges(etype)
        # self.input_feature = HeteroLinear.HeteroFeature({'protein': h_dict['protein']}, get_nodes_dict(self.g), embed_size=128, act=None)
        self.label_dict = self.load_label_dict(self.data_path.joinpath('label.pkl'))
        self.device = device
        self.batch_size = batch_size
        self.node_type = ['protein', 'go_annotation']
        self.category = 'protein'
        # protein_protein_sample_size = 5 
        # protein_go_sample_size = 3     
        # go_go_sample_size = 2            

        self.train_idx, self.valid_idx, self.test_idx = self.get_split('default')
        self.train_idx = self.train_idx.to(self.device)
        self.valid_idx = self.valid_idx.to(self.device)
        self.test_idx = self.test_idx.to(self.device)
        # self.remove_test_link()
        # self.g = self.add_edges_to_test_nodes(self.g, self.valid_idx)
        # self.g = self.add_edges_to_test_nodes(self.g, self.test_idx)
        
        '''
            meta_paths_dict: dict[str, list[etype]]
            Dict from meta path name to meta path.
        '''
        self.meta_paths_dict = {
            'meta_path_0': [('protein', 'interacts_0', 'go_annotation'), ('go_annotation', 'interacts_2', 'go_annotation'), ('go_annotation', '_interacts_0', 'protein')],
            'meta_path_1': [('protein', 'interacts_1', 'protein'), ('protein', 'interacts_1', 'protein'), ('protein', 'interacts_1', 'protein')],
            'meta_path_2': [('go_annotation', 'interacts_2', 'go_annotation'), ('go_annotation', 'interacts_2', 'go_annotation'), ('go_annotation', 'interacts_2', 'go_annotation')]
        }
        self.meta_paths = []
        for name, meta_path in self.meta_paths_dict.items():
            self.meta_paths.append(meta_path)
            
        meta_path_0 = [('protein', 'interacts_1', 'protein'), ('protein', 'interacts_0', 'go_annotation'), ('go_annotation', 'interacts_2', 'go_annotation')]

        self.sampler = dgl.dataloading.NeighborSampler([5])


        self.train_loader = dgl.dataloading.DataLoader(
                    self.g, {self.category: self.train_idx.to(self.device)}, self.sampler,
                    batch_size=self.batch_size, device=device, shuffle=True, drop_last=True)
        # , 'go_annotation':th.arange(0, self.go_num)
        self.valid_loader = dgl.dataloading.DataLoader(
                    self.g, {self.category: self.valid_idx.to(self.device)}, self.sampler,
                    batch_size=self.batch_size, device=device, shuffle=True, drop_last=True)
        self.test_loader = dgl.dataloading.DataLoader(
                    self.g, {self.category: self.test_idx.to(self.device)}, self.sampler,
                    batch_size=self.batch_size, device=device, shuffle=False, drop_last=False)


        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return 
    
    def get_features_list(self):
        features_list_path = self.data_path.joinpath('feature_list.pkl')
        if os.path.exists(features_list_path):
            features_list = pickle.load(open(features_list_path, 'rb'))
            return features_list
        else:
            features_list = []
            for i in range(len(self.nodes['count'])):
                th = self.nodes['attr'][i]
                if th is None:
                    features_list.append(ssp.eye(self.nodes['count'][i]))
                else:
                    features_list.append(th)
            pickle.dump(features_list, open(features_list_path, 'wb'))
            return features_list
    
    def get_e_feats(self):
        e_feats_path = self.data_path.joinpath('e_feats.pkl')
        if os.path.exists(e_feats_path):
            e_feats = pickle.load(open(e_feats_path, 'rb'))
            return e_feats
        else:
            edge2type = self.gen_edge2type(self)
            g = self.g
            e_feats = []
            for u, v in zip(*g.edges()):
                u = u.cpu().item()
                v = v.cpu().item()
                e_feats.append(edge2type[(u,v)])
            e_feats = th.tensor(e_feats, dtype=th.long).to(self.device)
            pickle.dump(e_feats, open(e_feats_path, 'wb'))
            return e_feats
        
    def gen_edge2type(self):
        edge2type = {}
        for k in self.links['data']:
            for u,v in zip(*self.links['data'][k].nonzero()):
                edge2type[(u,v)] = k
        for i in range(self.nodes['total']):
            if (i,i) not in edge2type:
                edge2type[(i,i)] = len(self.links['count'])
        for k in self.links['data']:
            for u,v in zip(*self.links['data'][k].nonzero()):
                if (v,u) not in edge2type:
                    edge2type[(v,u)] = k+1+len(self.links['count'])
        return edge2type
    

    def create_graph(self):
        data_path = self.data_path
        g_path = P.Path(self.g_path)
        if os.path.exists(g_path):
            g, _ = dgl.load_graphs(str(g_path))
            return g[0]
        else:
            node = pd.read_csv(data_path.joinpath('node.dat'), header=None, sep='\t')
            node.columns = ['id', 'name', 'type', 'feature']
            node['id'] = node.groupby('type').cumcount()
            node.loc[node['type'] == 1, 'feature'] = None
            output_path = str(data_path.joinpath('node.dat'))  # 目标文件路径
            node.to_csv(output_path, sep='\t', index=False, header=False)
            feature_dict = {'protein':[], 'go_annotation':[]}
            for _, row in node.iterrows():
                    if row['type'] == 0:
                        feature_dict['protein'].append(row['feature'].split(','))
                    if row['type'] == 1:
                        feature_dict['go_annotation'].append(row['feature'])
            protein_features = [list(map(float, feature)) for feature in feature_dict['protein']]
            protein_tensor = th.tensor(protein_features)
            emb_size = len(feature_dict['protein'][0])
            go_num = node[node['type'] == 1].shape[0]
            go_indices = torch.randint(0, go_num, (go_num,))
            feature_dict = {'protein': protein_tensor, 'go_annotation': F.one_hot(go_indices, num_classes=go_num)}

            protein_num = len(feature_dict['protein'])
            link = pd.read_csv(data_path.joinpath('link.dat'), header=None, sep='\t')
            link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
            link = pd.concat([link, link_test], axis=0)           
            p2g_src = link[link[2]==0][0].values
            p2g_des = link[link[2]==0][1].values
            p2p_src = link[link[2]==1][0].values
            p2p_des = link[link[2]==1][1].values
            g2g_src = link[link[2]==2][0].values
            g2g_des = link[link[2]==2][1].values
            # p2g_des = [go-protein_num for go in p2g_des]
            g2g_src = [go-protein_num for go in g2g_src]
            g2g_des = [go-protein_num for go in g2g_des]
            
            # edge_attrs_tensor = th.tensor(edge_attrs, dtype=th.float32)
            graph_data = {
                ('protein', 'interacts_0', 'go_annotation'): (th.tensor(p2g_src),th.tensor(p2g_des)),
                ('go_annotation', '_interacts_0', 'protein'): (th.tensor(p2g_des),th.tensor(p2g_src)),
                ('protein', 'interacts_1', 'protein'): (th.tensor(p2p_src),th.tensor(p2p_des)),
                ('protein', '_interacts_1', 'protein'): (th.tensor(p2p_des),th.tensor(p2p_src)),
                ('go_annotation', 'interacts_2', 'go_annotation'): (th.tensor(g2g_src),th.tensor(g2g_des)),
                ('go_annotation', '_interacts_2', 'go_annotation'): (th.tensor(g2g_des),th.tensor(g2g_src))
            }
            g = dgl.heterograph(graph_data)

            
            g.ndata['h'] = feature_dict
            for ntype in g.ntypes:
                g.nodes[ntype].data[dgl.NID] = torch.arange(g.num_nodes(ntype))
            # g = dgl.remove_self_loop(g)
            # g = dgl.add_self_loop(g)
            ppi_scores = link[link[2]==1][3].values
            g.edata['h'] = {
                ('protein', 'interacts_0', 'go_annotation'): th.ones(g.num_edges(etype='interacts_0'), 1,),
                ('go_annotation', '_interacts_0', 'protein'): th.ones(g.num_edges(etype='_interacts_0'), 1),
                ('protein', 'interacts_1', 'protein'): torch.tensor(ppi_scores, dtype=torch.float32),
                ('protein', '_interacts_1', 'protein'): torch.tensor(ppi_scores, dtype=torch.float32),
                ('go_annotation', 'interacts_2', 'go_annotation'):  th.ones(g.num_edges(etype='interacts_2'), 1),
                ('go_annotation', '_interacts_2', 'go_annotation'):  th.ones(g.num_edges(etype='_interacts_2'), 1)
            }
            for etype in g.edata['h']:
                g.edata['h'][etype] = g.edata['h'][etype].to(dtype=torch.float16)
            dgl.save_graphs(str(self.data_path.joinpath('graph.dgl')), g)
            return g

    def create_graph_dynamic(self, esm_name='esm1', top=100):
        data_path = self.data_path
        node = pd.read_csv(data_path.joinpath('node.dat'), header=None, sep='\t')
        node.columns = ['id', 'name', 'type', 'feature']
        node['id'] = node.groupby('type').cumcount()
        node.loc[node['type'] == 1, 'feature'] = None
        feature_dict = {'protein':[], 'go_annotation':[]}
        for _, row in node.iterrows():
                if row['type'] == 0:
                    feature_dict['protein'].append(row['feature'].split(','))
                if row['type'] == 1:
                    feature_dict['go_annotation'].append(row['feature'])
        protein_features = [list(map(float, feature)) for feature in feature_dict['protein']]
        protein_tensor = th.tensor(protein_features)
        go_num = node[node['type'] == 1].shape[0]
        go_indices = torch.randint(0, go_num, (go_num,))
        feature_dict = {'protein': protein_tensor, 'go_annotation': F.one_hot(go_indices, num_classes=go_num)}

        protein_num = len(feature_dict['protein'])
        # link = pd.read_csv(data_path.joinpath('link.dat'), header=None, sep='\t')
        # link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
        # link = pd.concat([link, link_test], axis=0) 
        
        terms_pkl = self.data_path.parent.joinpath('term.pkl')
        terms = pd.read_pickle(terms_pkl)
        # link_ori = pd.DataFrame([])
        # link_ori = link_ori[link_ori[2]==0]
        # link_1 = pd.read_csv('/root/autodl-tmp/source/pprogo-flg/data/{}_pp_{}.dat'.format(esm_name, top), header=None, sep='\t')
        # link_0 = pd.read_csv('/root/autodl-tmp/source/pprogo-flg/data/{}_pg_{}.dat'.format(esm_name, top), header=None, sep='\t')
        # link_0 = max(link_0, link_ori)
        link_0 = pd.read_csv(data_path.joinpath('esm1_mmseqs_pg_100.dat'), header=None, sep='\t')
        link_1 = pd.read_csv(data_path.parent.joinpath('esm1_mmseqs_pp_100.dat'), header=None, sep='\t')
        go_rels_csr = ssp.load_npz(self.data_path.parent.joinpath('go_rels_csr.npz'))
        link_2 = pd.read_pickle(self.data_path.joinpath('go_rel.pkl'))
        link_2.columns = [0,1,2,3]
        
        p2g_src = link_0[0].values
        p2g_des = link_0[1].values
        p2p_src = link_1[0].values
        p2p_des = link_1[1].values
        g2g_src = link_2[0].values
        g2g_des = link_2[1].values
        # p2g_des = [go-protein_num for go in p2g_des]
        g2g_src = [go-protein_num for go in g2g_src]
        g2g_des = [go-protein_num for go in g2g_des]
        
        # edge_attrs_tensor = th.tensor(edge_attrs, dtype=th.float32)
        graph_data = {
            ('protein', 'interacts_0', 'go_annotation'): (th.tensor(p2g_src),th.tensor(p2g_des)),
            ('go_annotation', '_interacts_0', 'protein'): (th.tensor(p2g_des),th.tensor(p2g_src)),
            ('protein', 'interacts_1', 'protein'): (th.tensor(p2p_src),th.tensor(p2p_des)),
            ('protein', '_interacts_1', 'protein'): (th.tensor(p2p_des),th.tensor(p2p_src)),
            ('go_annotation', 'interacts_2', 'go_annotation'): (th.tensor(g2g_src),th.tensor(g2g_des)),
            ('go_annotation', '_interacts_2', 'go_annotation'): (th.tensor(g2g_des),th.tensor(g2g_src))
        }
        g = dgl.heterograph(graph_data)

        g.ndata['h'] = feature_dict
        for ntype in g.ntypes:
            g.nodes[ntype].data[dgl.NID] = torch.arange(g.num_nodes(ntype))
            g.ndata[ntype]['_ID'] = torch.arange(g.num_nodes(ntype))
        # ppi_scores = link_1[3].values
        # g.edata['h'] = {
        #     ('protein', 'interacts_0', 'go_annotation'): th.ones(g.num_edges(etype='interacts_0'), 1,),
        #     ('go_annotation', '_interacts_0', 'protein'): th.ones(g.num_edges(etype='_interacts_0'), 1),
        #     ('protein', 'interacts_1', 'protein'): torch.tensor(ppi_scores, dtype=torch.float32),
        #     ('protein', '_interacts_1', 'protein'): torch.tensor(ppi_scores, dtype=torch.float32),
        #     ('go_annotation', 'interacts_2', 'go_annotation'):  th.ones(g.num_edges(etype='interacts_2'), 1),
        #     ('go_annotation', '_interacts_2', 'go_annotation'):  th.ones(g.num_edges(etype='_interacts_2'), 1)
        # }
        # for etype in g.edata['h']:
            # g.edata['h'][etype] = g.edata['h'][etype].to(dtype=torch.float16)
        # dgl.save_graphs(str(self.data_path.joinpath('graph_.dgl')), g)
        
        return g
    
    def get_split(self, mode='default', validation=True):
        r"""
        
        Parameters
        ----------
        validation : bool
            Whether to split dataset. Default ``True``. If it is False, val_idx will be same with train_idx.

        We can get idx of train, validation and test through it.

        return
        -------
        train_idx, val_idx, test_idx : torch.Tensor, torch.Tensor, torch.Tensor
        """
        if mode == 'default':
            link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
            test_idx = link_test.iloc[:, 0].drop_duplicates().values.tolist()
            all_nodes = list(range(self.g.number_of_nodes(self.category)))
            train_idx = [node for node in all_nodes if node not in test_idx]
            train_idx = th.tensor(train_idx, dtype=th.long)  # 转为 tensor 类型
            test_idx = th.tensor(test_idx, dtype=th.long)  

        elif mode == 'random':
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.go_indices)
            test_idx = th.tensor(test.ggo_indices)
        if validation:
            random_int = th.randperm(len(train_idx))
            valid_idx = train_idx[random_int[:len(train_idx) // 5]]
            train_idx = train_idx[random_int[len(train_idx) // 5:]]
        else:
            valid_idx = train_idx
            train_idx = train_idx

        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        # Here set test_idx as attribute of dataset to save results of HGB
        return self.train_idx, self.valid_idx, self.test_idx

    def load_label_dict(self, file_path):
        # 提取link和link_test中蛋白质-go的前两列，pid-gid
        label_path = self.data_path.joinpath('label.pkl')
        if os.path.exists(label_path):
            with open(file_path, 'rb') as f:
                label_dict = pickle.load(f)
        else:
            node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t')
            link = pd.read_csv(self.data_path.joinpath('link.dat'), header=None, sep='\t')  
            link = link[link[2]==0]
            link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
            protein_num = node[node[2]==0].shape[0]
            link[1] = link[1] - protein_num
            link_test[1] = link_test[1] - protein_num
            src = list(link[0]) + list(link_test[0])
            des = list(link[1]) + list(link_test[1])

            num = len(src)
            protein_num = self.protein_num
            go_num = node[node[2]==1].shape[0]

            label_protein = list(link[0]) + list(link_test[0])
            label_go = list(link[1]) + list(link_test[1])

            label_dict = {'protein':label_protein, 
                        'go':label_go}
            output_file = open(label_path, 'wb')
            pickle.dump(label_dict, output_file)

        # 将 label_dict['protein'] 的值与索引构建成一个字典，加快查找速度
        protein_to_go = {}
        for i, protein in enumerate(label_dict['protein']):
            if protein not in protein_to_go:
                protein_to_go[protein] = []
            protein_to_go[protein].append(label_dict['go'][i])

        return protein_to_go # 返回最大 go 编号作为类别数

    def get_label(self, protein_idx):
        # 初始化一个 (len(protein_idx), go_num) 的 label_list
        label_list = [[0] * self.go_num for _ in range(len(protein_idx))]
        
        # 加速查找，避免每次都访问 dict
        label_dict = self.label_dict
        
        # 映射 protein_idx 到 label_list 的索引，减少重复查找
        protein_idx_map = {idx: i for i, idx in enumerate(protein_idx)}

        # 遍历 label_dict 中的内容
        for idx in protein_idx:
            if idx in label_dict:
                for go in label_dict[idx]:
                    label_list[protein_idx_map[idx]][go] = 1  # 使用 protein_idx_map 映射加快速度

        return label_list
    
    def save_results(self, pred, label):
        node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t', low_memory=False)
        node.columns = ['id', 'name', 'type', 'feature']
        protein_node = node[node['type']==0]
        protein_name = protein_node['name']
        
        pred = pred.numpy()
        label = label.numpy()
        test_idx = list(self.test_idx.cpu())
        protein_name = protein_name[test_idx]
        
        np.savez(self.result_path, name=protein_name, labels=label, preds=pred)
        
    def get_remove_edge(self, node_ids, etype, form):

        if(form == 'out'):
            _, _, eids = self.g.out_edges(node_ids, etype=etype, form='all')
        elif(form == 'in'):
            _, _, eids = self.g.in_edges(node_ids, etype=etype, form='all')
        return eids
    
    def remove_test_link(self):
        etype_to_remove_1 = ('protein', 'interacts_0', 'go_annotation')
        etype_to_remove_2 = ('go_annotation', '_interacts_0', 'protein')
        valid_rm_edge_id_1 = self.get_remove_edge(self.valid_idx, etype_to_remove_1, 'out')
        valid_rm_edge_id_2 = self.get_remove_edge(self.valid_idx, etype_to_remove_2, 'in')
        test_rm_edge_id_1 = self.get_remove_edge(self.test_idx, etype_to_remove_1, 'out')
        test_rm_edge_id_2 = self.get_remove_edge(self.test_idx, etype_to_remove_2, 'in')
        self.g = dgl.remove_edges(self.g, torch.cat((valid_rm_edge_id_1,test_rm_edge_id_1)), etype=etype_to_remove_1)
        self.g = dgl.remove_edges(self.g, torch.cat((valid_rm_edge_id_2,test_rm_edge_id_2)), etype=etype_to_remove_2)
        
    def add_edges_to_test_nodes(self, hg, test_nodes):
        # 收集所有测试节点的一跳邻居及其对应的源节点
        first_hop_src_map = {}  # 键为（目标类型，目标节点），值为源节点列表
        target_etype = ('protein', 'interacts_0', 'go_annotation')
        target_etype_ = ('go_annotation', '_interacts_0', 'protein')
        
        for etype in hg.etypes:
            src_type, _, dst_type = hg.to_canonical_etype(etype)
            if src_type == 'protein':
                # 获取该边类型下测试节点的所有出边
                src, dst = hg.out_edges(test_nodes, etype=etype)
                src = src.cpu().numpy()
                dst = dst.cpu().numpy()
                for s, d in zip(src, dst):
                    key = (dst_type, d)
                    if key not in first_hop_src_map:
                        first_hop_src_map[key] = []
                    first_hop_src_map[key].append(s)

        # 收集二跳邻居中的目标节点并构建边
        second_hop_edges = []

        for (dst_type, d_node), src_nodes in first_hop_src_map.items():
            d_tensor = th.tensor([d_node], device=hg.device)
            for etype in hg.etypes:
                src_etype, _, dst_etype = hg.to_canonical_etype(etype)
                if src_etype == dst_type and dst_etype == 'go_annotation':
                    w_nodes = hg.successors(d_tensor, etype=etype)
                    if len(w_nodes) == 0:
                        continue
                    w_nodes = w_nodes.cpu().numpy().tolist()
                    for w in w_nodes:
                        for s in src_nodes:
                            second_hop_edges.append((s, w))

        # 去重并添加边
        if second_hop_edges:
            unique_edges = list({(s, w) for s, w in second_hop_edges})
            src_list, dst_list = zip(*unique_edges)
            src_nodes = th.tensor(src_list, device=hg.device)
            dst_nodes = th.tensor(dst_list, device=hg.device)
            hg.add_edges(src_nodes, dst_nodes, etype=target_etype)
            hg.add_edges(dst_nodes, src_nodes, etype=target_etype_)
        return hg
            
            

def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict

class random_walk_sampler(dgl.dataloading.Sampler):
    def __init__(self, g, num_randomwalk, device):
        self.meta_path_0 = [('protein', 'interacts_1', 'protein'), ('protein', 'interacts_1', 'protein')]
        self.meta_path_1 = [('protein', 'interacts_0', 'go_annotation'), ('go_annotation', 'interacts_2', 'go_annotation')]
        
        self.num_randomwalk = num_randomwalk
        self.device = device
        # self.sampler = dgl.sampling.RandomWalkNeighborSampler(g, 3, 0, num_neighbors, num_neighbors, self.meta_path_0)
        
    def sample(self, g, seeds, exclude_eids=None):
        # frontier = self.sampler(seeds['protein'])
        # subgraph = dgl.edge_subgraph(g, frontier.edges(), relabel_nodes=True)
        # isolated_nodes = torch.where(frontier.in_degrees() == 0)[0]
        # subgraph = dgl.remove_nodes(frontier, isolated_nodes)
        subgraph_protein_nodes = []
        subgraph_go_nodes = []
        for i in range(self.num_randomwalk):
            random_walk_nodes_0, types = dgl.sampling.random_walk(g, seeds['protein'], metapath=self.meta_path_0)
            random_walk_nodes_1, types = dgl.sampling.random_walk(g, seeds['protein'], metapath=self.meta_path_1)
            for i in range(seeds['protein'].shape[0]):
                subgraph_protein_nodes.append(random_walk_nodes_0[i][0])
                subgraph_protein_nodes.append(random_walk_nodes_0[i][1])

                subgraph_protein_nodes.append(random_walk_nodes_0[i][2])

                if(random_walk_nodes_1[i][1] == -1): continue
                subgraph_go_nodes.append(random_walk_nodes_1[i][1])
                if(random_walk_nodes_1[i][2] == -1): continue
                subgraph_go_nodes.append(random_walk_nodes_0[i][2])
        valid_protein_nodes = [node for node in subgraph_protein_nodes if node != -1]
        valid_go_nodes = [node for node in subgraph_go_nodes if node != -1]
        subgraph_nodes_dict = {
            'protein': torch.unique(torch.tensor(valid_protein_nodes)).to(self.device),
            'go_annotation': torch.unique(torch.tensor(valid_go_nodes).to(self.device))
        }
        
        subgraph = dgl.node_subgraph(g, subgraph_nodes_dict)
        return subgraph_nodes_dict, seeds, subgraph
        
        
