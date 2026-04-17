import dgl
import dgl.function as Fn
from dgl.ops import edge_softmax
from dgl.nn.pytorch import HeteroLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

class HGAT(nn.Module):
    """
        Parameters
        ----------
        num_layers: int
            the number of layers we used in the computing
        in_dim: int
            the input dimension
        hidden_dim: int
            the hidden dimension
        num_classes: int
            the number of the output classes
        ntypes: list
            the list of the node type in the graph
        negative_slope: float
            the negative slope used in the LeakyReLU
"""
    def __init__(self, ntypes, num_classes, feature_dim, hidden_dim=64, num_layers=2, negative_slope=0.2, dropout=0.8):
        super(HGAT, self).__init__()
        self.num_layers = num_layers
        self.activation = F.relu
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        # self.fc_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim)] for in_dim in in_dims)
        self.fc1 = nn.ModuleDict({
            # ntype: nn.Linear(feature_dim, hidden_dim)
            # for ntype in ntypes
            'protein': nn.Linear(feature_dim, hidden_dim),
            'go_annotation': nn.Linear(num_classes, hidden_dim)
        })
        self.output = nn.Linear(hidden_dim, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        nn.init.xavier_uniform_(self.fc1['protein'].weight)
        nn.init.xavier_uniform_(self.fc1['go_annotation'].weight)
        nn.init.xavier_uniform_(self.output.weight)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # self.input_bias = nn.Parameter(torch.zeros(hidden_dim))
            
        
        
        self.hgat_layers = nn.ModuleList()
        self.hgat_layers.append(
            TypeAttention(hidden_dim,
                            ntypes,
                            negative_slope))
        self.hgat_layers.append(
            NodeAttention(hidden_dim,
                            hidden_dim,
                            negative_slope)
        )
        for l in range(num_layers - 1):
            self.hgat_layers.append(
                TypeAttention(hidden_dim,
                            ntypes,
                            negative_slope))
            self.hgat_layers.append(
                NodeAttention(hidden_dim,
                            hidden_dim,
                            negative_slope)
            )
        
        # self.hgat_layers.append(
        #     TypeAttention(hidden_dim,
        #                     ntypes,
        #                     negative_slope))
        # self.hgat_layers.append(
        #     NodeAttention(hidden_dim,
        #                     num_classes,
        #                     negative_slope)
        # )
        
    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))
    
    def forward(self, hg, h_dict):
        """
        The forward part of the HGAT.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        with hg.local_scope():
            invalid_etypes = []
            for etype in hg.etypes:
                src, _ = hg.edges(etype=etype)
                if len(src) == 0:
                    invalid_etypes.append(etype)
            new_edges = {}
            for etypes3 in hg.canonical_etypes:
                if(etypes3[1] not in invalid_etypes):
                    new_edges[etypes3] = hg.edges(etype=etypes3)
            # new_edges = {
            #     etype: hg.edges(etype=etype) 
            #     for etype in hg.etypes 
            #     if etype not in invalid_etypes
            # }

            hg = dgl.heterograph(new_edges)
            
            for ntype, feat in h_dict.items():
                h_dict[ntype] = self.l2_norm(self.activation(self.fc1[ntype](feat.float())))
            hg.ndata['h'] = h_dict
            for l in range(self.num_layers):
                attention = self.hgat_layers[2 * l](hg, hg.ndata['h'])
                hg.edata['alpha'] = attention
                g = dgl.to_homogeneous(hg, ndata = 'h', edata = ['alpha'])
                h = self.hgat_layers[2 * l + 1](g, g.ndata['h'], g.ndata['_TYPE'], g.ndata['_TYPE'], presorted = True)
                if l == self.num_layers-1:
                    h = self.output(h)
                    # h = torch.sigmoid(h)

                h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
                hg.ndata['h'] = h_dict
        #     def _apply(ntype, h):
        #         # h = h + self.h_bias
        #         h = self.activation(h)
        #         h = self.bn(h)
        #         return self.dropout(h)
        # return {ntype: _apply(ntype, h) for ntype, h in h_dict.items()}
        return h_dict

class TypeAttention(nn.Module):
    """
    The type-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    ntypes: list
        the list of the node type in the graph
    slope: float
        the negative slope used in the LeakyReLU
    """
    def __init__(self, in_dim, ntypes, slope):
        super(TypeAttention, self).__init__()
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = in_dim
        self.mu_l = HeteroLinear(attn_vector, in_dim)
        self.mu_r = HeteroLinear(attn_vector, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        
    def forward(self, hg, h_dict):
        """
        The forward part of the TypeAttention.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        h_t = {}
        attention = {}
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                with rel_graph.local_scope():
                    degs = rel_graph.out_degrees().float().clamp(min = 1)
                    norm = torch.pow(degs, -0.5)
                    feat_src = h_dict[srctype]
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    feat_src = feat_src * norm
                    rel_graph.srcdata['h'] = feat_src
                    rel_graph.update_all(Fn.copy_u('h', 'm'), Fn.sum(msg='m', out='h'))
                    rst = rel_graph.dstdata['h']
                    degs = rel_graph.in_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    rst = rst * norm
                    h_t[srctype] = rst
                    h_l = self.mu_l(h_dict)[dsttype]
                    h_r = self.mu_r(h_t)[srctype]
                    edge_attention = F.elu(h_l + h_r)
                    # edge_attention = F.elu(h_l + h_r).unsqueeze(0)
                    # rel_graph.ndata['m'] = {dsttype: edge_attention,
                    #                 srctype: torch.zeros((rel_graph.num_nodes(ntype = srctype),)).to(edge_attention.device)}
                    if srctype == dsttype:
                        rel_graph.ndata['m'] = edge_attention 
                    else:
                        rel_graph.ndata['m'] = {dsttype: edge_attention,
                                            srctype: torch.zeros((rel_graph.num_nodes(ntype=srctype),)).to(edge_attention.device)}
                    reverse_graph = dgl.reverse(rel_graph)
                    reverse_graph.apply_edges(Fn.copy_u('m', 'alpha'))
                hg.edata['alpha'] = {(srctype, etype, dsttype): reverse_graph.edata['alpha']}
                
                # if dsttype not in attention.keys():
                #     attention[dsttype] = edge_attention
                # else:
                #     attention[dsttype] = torch.cat((attention[dsttype], edge_attention))
            attention = edge_softmax(hg, hg.edata['alpha'])
            # for ntype in hg.dsttypes:
            #     attention[ntype] = F.softmax(attention[ntype], dim = 0)

        return attention
    
class NodeAttention(nn.Module):
    """
    The node-level attention layer

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    out_dim: int
        the output dimension
    slope: float
        the negative slope used in the LeakyReLU
    """
    def __init__(self, in_dim, out_dim, slope):
        super(NodeAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Mu_l = nn.Linear(in_dim, in_dim)
        self.Mu_r = nn.Linear(in_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        
    def forward(self, g, x, ntype, etype, presorted = False):
        """
        The forward part of the NodeAttention.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``
            
        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        with g.local_scope():
            src = g.edges()[0]
            dst = g.edges()[1]
            h_l = self.Mu_l(x)[src]
            h_r = self.Mu_r(x)[dst]
            edge_attention = self.leakyrelu((h_l + h_r) * g.edata['alpha'])
            edge_attention = edge_softmax(g, edge_attention)
            g.edata['alpha'] = edge_attention
            g.srcdata['x'] = x
            g.update_all(Fn.u_mul_e('x', 'alpha', 'm'),
                         Fn.sum('m', 'x'))
            h = g.ndata['x']
        return h
    

     # def hgat(num_layers, 
     #      hidden_dims,
     #      num_classes, **kwargs):
     #      return hg.HGAT(num_layers=num_layers, 
     #                     hidden_dims=hidden_dims, 
     #                     num_classes=num_classes, **kwargs)

def to_hetero_feat(h, type, name):
    """Feature convert API.

    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.

    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.

    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph

    Example
    -------

    >>> h = torch.tensor([[1, 2, 3],
                          [1, 1, 1],
                          [0, 2, 1],
                          [1, 3, 3],
                          [2, 1, 1]])
    >>> print(h.shape)
    torch.Size([5, 3])
    >>> type = torch.tensor([0, 1, 0, 0, 1])
    >>> name = ['author', 'paper']
    >>> h_dict = to_hetero_feat(h, type, name)
    >>> print(h_dict)
    {'author': tensor([[1, 2, 3],
    [0, 2, 1],
    [1, 3, 3]]), 'paper': tensor([[1, 1, 1],
    [2, 1, 1]])}

    """
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[torch.where(type == index)]

    return h_dict
