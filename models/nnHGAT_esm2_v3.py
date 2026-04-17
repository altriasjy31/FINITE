import dgl
import dgl.function as Fn
from dgl.ops import edge_softmax
from dgl.nn.pytorch import HeteroLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

def to_hetero_feat(h_homo: torch.Tensor, type_ids: torch.Tensor, ntypes):
    """Split homogeneous node features back to hetero dict by _TYPE."""
    out = {}
    for t, ntype in enumerate(ntypes):
        idx = (type_ids == t).nonzero(as_tuple=True)[0]
        out[ntype] = h_homo[idx]
    return out


class TypeAttention(nn.Module):
    """
    Type-level attention on a heterograph (NOT block).
    Output alpha per edge is scalar: (E_rel, 1).
    Per-relation edge_softmax for stability.
    """

    def __init__(self, hidden_dim: int, ntypes, slope: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ntypes = list(ntypes)

        self.mu_l = HeteroLinear({nt: hidden_dim for nt in self.ntypes}, hidden_dim)
        self.mu_r = HeteroLinear({nt: hidden_dim for nt in self.ntypes}, hidden_dim)

        # reduce dst message to scalar attention logit
        self.attn_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(slope)

    def forward(self, hg_graph: dgl.DGLHeteroGraph, h_dict: dict):
        """
        hg_graph: heterograph produced by dgl.block_to_graph(block)
        h_dict: hg_graph.ndata['h'] dict, each (N_type, hidden_dim)
        return: alpha_dict[(srctype, etype, dsttype)] = (E_rel, 1)
        """
        with hg_graph.local_scope():
            dev = hg_graph.device
            alpha_dict = {}

            for (srctype, etype, dsttype) in hg_graph.canonical_etypes:
                rel = hg_graph[srctype, etype, dsttype]
                E = rel.num_edges()

                if E == 0:
                    alpha_dict[(srctype, etype, dsttype)] = torch.zeros((0, 1), device=dev)
                    continue

                if (srctype not in h_dict) or (dsttype not in h_dict):
                    alpha_dict[(srctype, etype, dsttype)] = torch.zeros((E, 1), device=dev)
                    continue

                with rel.local_scope():
                    # --- normalized src -> dst aggregation to build "rst" on dst side ---
                    feat_src = h_dict[srctype]  # (N_src, D)
                    rel.srcdata["h_src"] = feat_src

                    out_deg = rel.out_degrees().float().clamp(min=1)
                    norm_out = torch.pow(out_deg, -0.5).unsqueeze(-1).to(feat_src.device)
                    rel.srcdata["h_src"] = rel.srcdata["h_src"] * norm_out

                    rel.update_all(Fn.copy_u("h_src", "m"), Fn.sum("m", "rst"))
                    rst = rel.dstdata["rst"]  # (N_dst, D)

                    in_deg = rel.in_degrees().float().clamp(min=1)
                    norm_in = torch.pow(in_deg, -0.5).unsqueeze(-1).to(rst.device)
                    rst = rst * norm_in

                    # --- compute dst-side message & edge logits ---
                    h_dst = h_dict[dsttype]  # (N_dst, D)

                    # nnHGAT_esm2_v3.py :: TypeAttention.forward

                    z_l = self.mu_l({dsttype: h_dst})[dsttype]   # (N_dst, D)
                    # z_r = self.mu_r({srctype: rst})[srctype]     # ❌ 实际是 (N_src, D)
                    z_r = self.mu_r({dsttype: rst})[dsttype]
                    m_dst = F.elu(z_l + z_r)

                    logit_dst = self.attn_proj(m_dst)            # (N_dst, 1)
                    logit_dst = self.leakyrelu(logit_dst)

                    rel.dstdata["logit"] = logit_dst
                    rel.apply_edges(lambda e: {"elogit": e.dst["logit"]})  # (E,1)

                    # per-relation edge softmax; force fp32 for stability
                    alpha = edge_softmax(rel, rel.edata["elogit"].float())  # (E,1)

                alpha_dict[(srctype, etype, dsttype)] = alpha

            hg_graph.edata["alpha"] = alpha_dict
            return alpha_dict


class NodeAttention(nn.Module):
    """
    Node-level attention on homogeneous graph using scalar edge weights alpha (E,1).
    """

    def __init__(self, hidden_dim: int, slope: float = 0.2):
        super().__init__()
        self.Mu_l = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Mu_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU(slope)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor):
        with g.local_scope():
            src, dst = g.edges()
            h_l = self.Mu_l(x)[src]   # (E, D)
            h_r = self.Mu_r(x)[dst]   # (E, D)

            alpha = g.edata["alpha"]
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(-1)  # (E,1)

            # edge score scalar
            e = self.leakyrelu((h_l + h_r) * alpha)  # (E,D)
            e = e.sum(dim=1, keepdim=True)           # (E,1)

            a = edge_softmax(g, e)                   # (E,1)
            g.edata["a"] = a

            g.srcdata["x"] = x
            g.update_all(Fn.u_mul_e("x", "a", "m"), Fn.sum("m", "x"))
            return g.ndata["x"]


class HGAT_ESM2_V3(nn.Module):
    """
    Inductive HGAT with fusion (profunsom/msa + esm2).
    - 输入 block（blocks[-1]）
    - h_src：blocks[0].srcdata['h']（base ntype: protein/go_annotation）
    - esm2_p/esm2_g：按 blocks[0].srcnodes[*].data[dgl.NID] gather 得到的 (N_src_type, 1280)
    - 输出：{"protein": logits} 其中 logits shape=(N_dst_protein, go_num)
    """

    def __init__(
        self,
        ntypes,
        num_classes,
        protein_feat_dim,   # 2048
        go_feat_dim,        # 2048 or num_classes depending on your dataset
        hidden_dim=512,
        num_layers=2,
        slope=0.2,
        dropout=0.5,
        use_esm2=True,
        esm2_protein_dim=1280,
        esm2_go_dim=1280,
    ):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.ntypes = list(ntypes)  # ['protein','go_annotation']
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_esm2 = use_esm2
        half = hidden_dim // 2

        # ---- fusion projection ----
        self.proj_msa = nn.ModuleDict({
            "protein": nn.Linear(protein_feat_dim, half),
            "go_annotation": nn.Linear(go_feat_dim, half),
        })
        self.proj_esm2 = nn.ModuleDict({
            "protein": nn.Linear(esm2_protein_dim, half, bias=False),
            "go_annotation": nn.Linear(esm2_go_dim, half, bias=False),
        }) if use_esm2 else None

        self.fuse_ln = nn.ModuleDict({
            "protein": nn.LayerNorm(hidden_dim),
            "go_annotation": nn.LayerNorm(hidden_dim),
        })
        self.fuse_drop = nn.Dropout(dropout)

        # ---- HGAT layers ----
        self.type_attn = nn.ModuleList([
            TypeAttention(hidden_dim, self.ntypes, slope)
            for _ in range(num_layers)
        ])
        self.node_attn = nn.ModuleList([
            NodeAttention(hidden_dim, slope)
            for _ in range(num_layers)
        ])

        # ---- head on protein dst ----
        self.head_ln = nn.LayerNorm(hidden_dim)
        self.head_drop = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden_dim, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _fuse(self, ntype, x_msa, x_esm2=None):
        msa_h = self.proj_msa[ntype](x_msa.float())  # (N, half)
        if self.use_esm2 and (x_esm2 is not None) and (self.proj_esm2 is not None):
            esm_h = self.proj_esm2[ntype](x_esm2.float())
        else:
            esm_h = torch.zeros_like(msa_h)
        h = torch.cat([msa_h, esm_h], dim=1)  # (N, hidden)
        h = self.fuse_ln[ntype](h)
        h = self.fuse_drop(h)
        return h

    def forward(self, g, h_msa_dict, esm2_dict=None):
        """
        g: dgl.DGLHeteroGraph 子图（node_subgraph 得到）
        h_msa_dict: g.ndata['h']，每个 ntype: (N_type, 256)
        esm2_dict: 可选，{ 'protein': (N_p,1280), 'go_annotation': (N_go,1280) }
        """
        # 1) fuse to hidden_dim(512)
        h = {}
        for ntype in g.ntypes:
            x_msa = h_msa_dict[ntype]              # (N,256)
            x_esm2 = None
            if esm2_dict is not None and ntype in esm2_dict:
                x_esm2 = esm2_dict[ntype]          # (N,1280)
            h[ntype] = self._fuse(ntype, x_msa, x_esm2)  # (N,512)

        # 2) HGAT layers on heterograph (NOT block)
        for l in range(self.num_layers):
            alpha_dict = self.type_attn[l](g, h)

            with g.local_scope():
                g.ndata["h"] = h
                g.edata["alpha"] = alpha_dict

                gh = dgl.to_homogeneous(g, ndata=["h"], edata=["alpha"])
                x = gh.ndata["h"]
                # gh.edata["alpha"] already exists
                x_out = self.node_attn[l](gh, x)

                # split back to hetero
                h = to_hetero_feat(x_out, gh.ndata["_TYPE"], g.ntypes)

        # 3) head on protein (return all proteins in this subgraph)
        hp = h["protein"]
        logits = self.cls_head(self.head_drop(self.head_ln(hp)))
        return {"protein": logits}
