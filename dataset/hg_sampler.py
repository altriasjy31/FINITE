import dgl
import torch
from dgl.dataloading import Sampler, DataLoader
import time

# 文件：hg_sampler.py
# 作用：定义 hg_Sampler 类，用于在异构图上采样子图，支持小批量训练
# 功能：
#   - 采样蛋白质-蛋白质（interacts_1）、蛋白质-GO（interacts_0）和 GO-GO（interacts_2）关系的邻居
#   - 合并采样子图为完整的异构子图，供模型训练
#   - 支持自定义采样数量（num_protein_protein、num_protein_go、num_go_go）
# 数据集：依赖 hg_dataset.py 或 inductive_learning_dataset.py 提供的异构图
# 使用方法：
#   - 初始化：hg_Sampler(g, num_protein_protein=5, num_protein_go=3, num_go_go=2)
#   - 在 DataLoader 中调用：dgl.dataloading.DataLoader(g, seeds, sampler)
#   - 运行示例：与 hgat_trainer.py 配合使用

# 数据集：依赖 hg_dataset.py 或 inductive_learning_dataset.py 的异构图。

def get_edge_subgraph(g, sample_subgraph):
    all_edges = {}
    for etype in sample_subgraph.canonical_etypes:
        src, dst = sample_subgraph.edges(etype=etype)
        all_edges[etype] = (src, dst)

    # 合并所有边的源节点和目标节点，并为每种类型去重
    subgraph_nodes = {}
    for etype in all_edges:
        src, dst = all_edges[etype]
        subgraph_nodes[etype] = torch.cat([src, dst]).unique()

    # 创建一个新的子图，只包含采样边及其连接的节点
    subgraph = dgl.edge_subgraph(g, subgraph_nodes)
    
    return subgraph
    
class hg_Sampler(Sampler):
    def __init__(self, g, num_protein_protein, num_protein_go, num_go_go):
        super().__init__()
        self.g = g
        self.num_protein_protein = num_protein_protein
        self.num_protein_go = num_protein_go
        self.num_go_go = num_go_go

    def sample(self, g, batch_nodes):
        time_0 = time.time()
        batch_nodes = batch_nodes['protein']
        # 采样蛋白质-蛋白质关系
        protein_protein = dgl.sampling.sample_neighbors(
            self.g, {'protein':batch_nodes}, 
            fanout={('protein', 'interacts_0', 'go_annotation'): 0,
                ('go_annotation', '_interacts_0', 'protein'): 0,
                ('protein', 'interacts_1', 'protein'): self.num_protein_protein,
                ('protein', '_interacts_1', 'protein'): self.num_protein_protein,
                ('go_annotation', 'interacts_2', 'go_annotation'): 0,
                ('go_annotation', '_interacts_2', 'go_annotation'): 0},
            edge_dir='in'
        )
        protein_protein_subgraph = get_edge_subgraph(g, protein_protein)
        
        # 采样蛋白质-GO标签关系
        protein_go = dgl.sampling.sample_neighbors(
            self.g, {'protein':batch_nodes}, 
            fanout={('protein', 'interacts_0', 'go_annotation'): self.num_protein_go,
                ('go_annotation', '_interacts_0', 'protein'): 0,
                ('protein', 'interacts_1', 'protein'): 0,
                ('protein', '_interacts_1', 'protein'): 0,
                ('go_annotation', 'interacts_2', 'go_annotation'): 0,
                ('go_annotation', '_interacts_2', 'go_annotation'): 0},
            edge_dir='in'
        )
        protein_go_subgraph = get_edge_subgraph(g, protein_go)
        
        
        # 采样GO-GO关系
        # 首先找到所有与蛋白质节点相关的GO节点
        go_nodes = protein_go_subgraph.ndata[dgl.NID]['go_annotation']
        go_go = dgl.sampling.sample_neighbors(
            self.g, {'go_annotation':go_nodes}, 
            fanout={('protein', 'interacts_0', 'go_annotation'): 0,
                ('go_annotation', '_interacts_0', 'protein'): 0,
                ('protein', 'interacts_1', 'protein'): 0,
                ('protein', '_interacts_1', 'protein'): 0,
                ('go_annotation', 'interacts_2', 'go_annotation'): self.num_go_go,
                ('go_annotation', '_interacts_2', 'go_annotation'): self.num_go_go},
            edge_dir='in'
        )
        go_go_subgraph = get_edge_subgraph(g, go_go)
        # 合并所有采样的子图
        # subg = dgl.merge([protein_protein_subgraph, protein_go_subgraph, go_go_subgraph])

        # protein_original_ids = subg.ndata[dgl.NID]['protein']
        # # 获取子图中所有蛋白质节点的原始 ID
        # return batch_nodes, protein_original_ids, subg
        protein_protein_ids = protein_protein_subgraph.ndata[dgl.NID]['protein']
        protein_go_ids = protein_go_subgraph.ndata[dgl.NID]['go_annotation']
        go_go_ids = go_go_subgraph.ndata[dgl.NID]['go_annotation']
        
        # 合并所有的子图，并确保拼接是根据原始图中的节点ID
        # all_node_ids = torch.cat([protein_protein_ids, protein_go_ids, go_go_ids])
        all_node_ids = {'protein':torch.cat([protein_protein_ids, batch_nodes]), 'go_annotation':torch.cat([protein_go_ids, go_go_ids])}
        all_protein_ids, protein_unique_idx = torch.unique(all_node_ids['protein'], return_inverse=True)
        all_go_ids, go_unique_idx = torch.unique(all_node_ids['go_annotation'], return_inverse=True)
        all_node_ids = {'protein': all_protein_ids, 'go_annotation': all_go_ids}
        # 手动合并子图，确保按照原始 ID 进行拼接
        subg = dgl.node_subgraph(self.g, all_node_ids)

        # 获取子图中所有蛋白质节点的原始 ID
        protein_original_ids = subg.ndata[dgl.NID]['protein']

        return batch_nodes, protein_original_ids, subg