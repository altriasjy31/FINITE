#!/usr/bin/env python3

"""
HGAT训练脚本（ESM2 + ProFun-SOM特征融合版本，V3）

功能：
- 使用 HGAT_ESM2_V3 模型，融合 ESM2 和 ProFun-SOM 特征（相加式 + LayerNorm）
- 保存模型到 models/hgat_esm2_inductive{bp,cc,mf}_v3
- 保存预测结果到 my_results/inductive_HGAT_ESM2_{bp,cc,mf}_v3.npz
- V3 版本：在 V2（无 bias + 可学习权重）的基础上，引入相加式融合 + LayerNorm，减轻尺度不匹配带来的训练不稳定
"""
import os
import sys
import time
import argparse
import pathlib as P
from pathlib import Path
import json
prj_root = str(P.Path(__file__).parent.parent)
if prj_root not in sys.path:
    sys.path.append(prj_root)

# from util.utils import EarlyStopping
# from models.HGAT_esm2_v3 import HGAT_ESM2_V3
# from dataset.inductive_learning_dataset import DBLPDataset
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # torch.cuda.empty_cache()
# import torch.optim as optim
# from torch.amp import autocast, GradScaler
# from torch.optim.lr_scheduler import OneCycleLR
# import dgl
# # from dgl.sampling import RandomWalkNeighborSampler
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import precision_recall_curve, f1_score, auc
# # import concurrent.futures
# import warnings
# from sklearn.exceptions import UndefinedMetricWarning

# warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", category=UserWarning)

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--go_name', type=str, default='bp', choices=['bp', 'cc', 'mf'])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--dataset_sampler_type', type=str, default='hgat', choices=['hgat', 'graphsage'],
                        help='InductiveLearningDataset sampler type.'+\
                            'hgat: single layer fanout; graphsage: multi-layer fanouts')
    parser.add_argument('--dataset_fanouts', type=str, default='5,5',
                        help='specify fanouts for each layer.')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_pct_start', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--ema', action='store_true', help='Use EMA model for evaluation.')

    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:2')
    return parser

def main():
    # 1) 先 parse args —— 这样 --help 会在这里直接退出，不会触发任何重型 import
    parser = build_parser()
    args = parser.parse_args()

    # 2) 只有真正运行训练时，才做重型 import（torch/dgl/sklearn 等）
    # 可选：限制线程池，避免退出阶段卡住（尤其是 sklearn/numpy/torch 后端）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.amp import autocast, GradScaler
    from torch.optim.lr_scheduler import OneCycleLR
    import dgl
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from sklearn.metrics import precision_recall_curve, auc

    # 你的工程路径注入也建议放这里，避免 help 时动工程环境
    prj_root = str(P.Path(__file__).parent.parent)
    if prj_root not in sys.path:
        sys.path.append(prj_root)

    from util.utils import EarlyStopping
    # from models.HGAT_esm2_v3 import HGAT_ESM2_V3
    from models.nnHGAT_esm2_v3 import HGAT_ESM2_V3
    from dataset.inductive_learning_dataset import DBLPDataset
    from util.loss import AsymmetricLossOptimized
    from util.metrics import fmax_score, f1_max
    from util.moving_average import ModelEma
    from util.utils import parse_fanouts

    # CUDA cache 清理：不建议在 import 阶段执行；需要的话放到这里
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    saving_dir = P.Path(f"/data0/shaojiangyi/pprogo-flg-2/models/{args.go_name}_{run_id}")
    saving_dir.mkdir(parents=True, exist_ok=True)

    trainer_args = {
        'go_name': args.go_name,
        'epoch': args.epoch,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dataset_sampler_type': args.dataset_sampler_type,
        'dataset_fanouts': parse_fanouts(args.dataset_fanouts),
        'lr': args.lr,
        "lr_pct_start": args.lr_pct_start,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'ema': args.ema,
        'device': args.device,
        'model_path': saving_dir / f'hgat_esm2_inductive{args.go_name}_v3',
    }

    class HGAT_ESM2_V3_Trainer(object):

        def __init__(self, args):
            self.lr = args['lr']
            self.weight_decay = args['weight_decay']
            self.device = args['device']
            self.dataset = DBLPDataset('data', args['go_name'], 'HGAT', args['batch_size'], self.device,
                                       sampler_type=args['dataset_sampler_type'],
                                       dataset_fanouts=args['dataset_fanouts'])

            self.g = self.dataset.g.to(self.device)

            # 加载 ESM2 特征
            self.esm2_protein_features, self.esm2_go_features = self.load_esm2_features(args['go_name'])

            # 从 node.dat 加载蛋白质名称和 GO 术语名称
            self.protein_name_dict, self.go_terms_list = self.load_node_names(args['go_name'])
            self.ema_model = None

            # 使用 HGAT_ESM2_V3 模型（保持 HGAT 主体结构不变，只改融合层）
            self.model = HGAT_ESM2_V3(
                self.dataset.node_type,
                num_classes=self.dataset.go_num,
                protein_feat_dim=self.dataset.feature_dim,
                go_feat_dim=getattr(self.dataset, 'go_feature_dim', self.dataset.feature_dim),
                hidden_dim=args.get('hidden_dim', 512),
                num_layers=args.get('num_layers', 2),
                use_esm2=True,
                esm2_protein_dim=1280,
                esm2_go_dim=1280,
            )
            if args.get('ema', False):
                self.ema_model = ModelEma(self.model, decay=0.997, device=self.device)
            self.model = self.model.to(self.device)
            self.model_path = args['model_path']
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr,
                                         weight_decay=self.weight_decay,
                                         eps=1e-6)
            self.best_fmax = 0
            self.epoch = args['epoch']
            self.batch_size = args['batch_size']
            self.patience = args['patience']
            # self.loss_fn = nn.BCEWithLogitsLoss()
            self.loss_fn = AsymmetricLossOptimized(gamma_pos=1, gamma_neg=4, clip=0.05)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, pct_start=args["lr_pct_start"],total_steps=self.epoch * len(self.dataset.train_loader))
            self.category = 'protein'
            self.meta_paths_dict = {
                'mp_0': [('protein', 'interacts_0', 'go_annotation')],
                'mp_1': [('protein', 'interacts_1', 'protein')],
                'mp_2': [('go_annotation', 'interacts_2', 'go_annotation')]
            }
            # ---- best-by-metrics saving ----
            self.best_fmax = -1.0
            self.best_auprc = -1.0

            # two separate checkpoints
            self.best_fmax_path = self.model_path.parent / (self.model_path.name + "_bestFmax.pt")
            self.best_auprc_path = self.model_path.parent / (self.model_path.name + "_bestAUPRC.pt")

            # optional: record what epoch achieved the best
            self.best_fmax_epoch = -1
            self.best_auprc_epoch = -1

            # ---- best-both / pareto saving ----
            self.best_both_score = -1e18
            self.best_both_epoch = -1
            self.best_both_path = self.model_path.parent / (self.model_path.name + "_bestBoth.pt")

            # strict "both improved" best (optional, very strict)
            self.best_both_strict_fmax = -1.0
            self.best_both_strict_auprc = -1.0
            self.best_both_strict_epoch = -1
            self.best_both_strict_path = self.model_path.parent / (self.model_path.name + "_bestBothStrict.pt")

            # pareto front cache: list of dicts [{"epoch":..., "fmax":..., "auprc":..., "path":...}, ...]
            self.pareto_front = []
            self.pareto_dir = self.model_path.parent / "pareto_ckpts"
            self.pareto_dir.mkdir(parents=True, exist_ok=True)

        def load_esm2_features(self, go_name: str):
            """Load ESM2 features (protein/go). Return (protein_feat, go_feat) tensors on self.device."""
            esm2_dir = P.Path(f"/data0/shaojiangyi/pprogo-flg-2/data/{go_name}/esm2_features")
            protein_path = esm2_dir / "esm2_protein_features.npy"
            go_path = esm2_dir / "esm2_go_features.npy"

            esm2_protein_features = None
            esm2_go_features = None

            if protein_path.exists():
                arr = np.load(protein_path)
                esm2_protein_features = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
                print(f"[INFO] Loaded ESM2 protein features: {protein_path} shape={tuple(esm2_protein_features.shape)}")
            else:
                print(f"[WARN] Missing ESM2 protein features: {protein_path}")

            if go_path.exists():
                arr = np.load(go_path)
                esm2_go_features = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
                print(f"[INFO] Loaded ESM2 GO features: {go_path} shape={tuple(esm2_go_features.shape)}")
            else:
                print(f"[WARN] Missing ESM2 GO features: {go_path}")

            return esm2_protein_features, esm2_go_features


        def load_node_names(self, go_name: str):
            """Load protein names and GO term names from node.dat."""
            node_path = P.Path(f"/data0/shaojiangyi/pprogo-flg-2/data/{go_name}/node.dat")
            node = pd.read_csv(node_path, header=None, sep="\t", dtype={3: str})
            node.columns = ["id", "name", "type", "feature"]

            protein_nodes = node[node["type"] == 0].reset_index(drop=True)
            protein_name_dict = {i: n for i, n in enumerate(protein_nodes["name"].tolist())}

            go_nodes = node[node["type"] == 1].reset_index(drop=True)
            go_terms_list = go_nodes["name"].tolist()

            print(f"[INFO] Node names loaded: protein={len(protein_name_dict)} go={len(go_terms_list)}")
            return protein_name_dict, go_terms_list


        def _gather_block_features(self, blocks):
            """取 blocks[0] 的 src 节点特征字典（ProFun-SOM / GO embedding）"""
            h_dict = {}
            for ntype in blocks[0].ntypes:
                # 注意：如果你把特征存成 blocks[0].srcdata['h'][ntype] 就这样取
                h_dict[ntype] = blocks[0].srcdata['h'][ntype]
            return h_dict

        def _gather_esm2_by_parent_nid(self, blocks):
            """
            用 parent NID（dgl.NID）去全局 ESM2 特征矩阵 gather，
            严格避免使用 input_nodes（那是 subgraph local id，和 parent id 不一致）。
            """
            esm2_protein_feat = None
            esm2_go_feat = None

            if self.esm2_protein_features is not None and 'protein' in blocks[0].srctypes:
                protein_parent = blocks[0].srcnodes['protein'].data[dgl.NID]
                if protein_parent.numel() > 0:
                    esm2_protein_feat = self.esm2_protein_features[protein_parent]

            if self.esm2_go_features is not None and 'go_annotation' in blocks[0].srctypes:
                go_parent = blocks[0].srcnodes['go_annotation'].data[dgl.NID]
                if go_parent.numel() > 0:
                    esm2_go_feat = self.esm2_go_features[go_parent]

            return esm2_protein_feat, esm2_go_feat

        def _labels_for_dst_proteins(self, blocks):
            """
            训练/验证标签都按 blocks[-1].dst 的 protein parent id 取，
            这与 dataset.get_label(protein_parent_ids) 一致。
            """
            if 'protein' not in blocks[-1].dsttypes:
                return None, None

            dst_parent = blocks[-1].dstnodes['protein'].data[dgl.NID]  # parent ids
            if dst_parent.numel() == 0:
                return None, None

            dst_parent_list = dst_parent.detach().cpu().tolist()
            labels_list = self.dataset.get_label(dst_parent_list)  # expects parent ids
            labels = torch.tensor(labels_list, dtype=torch.float32, device=self.device)
            return dst_parent, labels

        def _build_sg_and_feats(self, base_g, input_nodes):
            sg = dgl.node_subgraph(base_g, input_nodes)   # input_nodes 是 base_g id space
            h_msa = sg.ndata["h"]

            esm2 = {}
            if self.esm2_protein_features is not None and sg.num_nodes("protein") > 0:
                p_parent = sg.nodes["protein"].data["parent_nid"]   # full_g id space
                esm2["protein"] = self.esm2_protein_features[p_parent]

            if self.esm2_go_features is not None and sg.num_nodes("go_annotation") > 0:
                g_parent = sg.nodes["go_annotation"].data["parent_nid"]
                esm2["go_annotation"] = self.esm2_go_features[g_parent]

            return sg, h_msa, esm2

        def _map_output_to_sg(self, sg, ntype, output_ids):
            # output_ids: Tensor in base_g id space
            sg_parent = sg.nodes[ntype].data[dgl.NID]  # base_g ids, aligned with sg local order
            pos = {int(pid): i for i, pid in enumerate(sg_parent.detach().cpu().tolist())}
            idx = torch.tensor([pos[int(x)] for x in output_ids.detach().cpu().tolist()],
                            device=output_ids.device, dtype=torch.long)
            return idx



        def train(self):
            model = self.model
            stopper = EarlyStopping(self.patience, self.model_path)

            amp = self.device.startswith("cuda") and torch.cuda.is_available()
            scaler = GradScaler(enabled=amp)

            grad_clip = 1.0  # 建议先开着，后续你可调 0.5/2.0
            for epoch in range(self.epoch):
                model.train()
                total_loss = 0.0
                n_batches = 0

                for input_nodes, output_nodes, blocks in tqdm(self.dataset.train_loader, desc=f'Epoch {epoch+1}/{self.epoch}'):
                    self.optimizer.zero_grad(set_to_none=True)

                    sg, h_msa, esm2 = self._build_sg_and_feats(self.dataset.train_g, input_nodes)

                    # forward: logits for ALL proteins in sg
                    out = self.model(sg, h_msa, esm2)
                    logits_all = out["protein"]

                    if "protein" not in output_nodes:
                        continue
                    out_p = output_nodes["protein"]  # base_g(train_g) id space

                    # map output proteins to sg-local indices
                    idx = self._map_output_to_sg(sg, "protein", out_p)
                    logits = logits_all[idx]         # (B, go_num)

                    # labels: 必须用 full_g 原始 protein id（parent_nid）
                    out_parent_full = self.dataset.train_g.nodes["protein"].data["parent_nid"][out_p]
                    labels = torch.tensor(self.dataset.get_label(out_parent_full.detach().cpu().tolist()),
                          dtype=torch.float32, device=self.device)

                    loss = self.loss_fn(logits, labels)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    if self.ema_model is not None:
                        self.ema_model.update(model)

                    total_loss += float(loss.item())
                    n_batches += 1

                avg_loss = total_loss / max(n_batches, 1)
                print(f"Epoch {epoch+1}/{self.epoch}, Loss: {avg_loss:.4f}")

                val_fmax, val_aupr, val_loss = self.evaluate(self.dataset.valid_loader, 'Valid')
                print(f'Valid - Fmax: {val_fmax:.4f}, AUPR: {val_aupr:.4f}, Loss: {val_loss:.4f}')

                self.save_best_by_metrics(epoch+1, val_fmax, val_aupr, extra={"val_loss": float(val_loss)})
                self.save_best_both(epoch+1, val_fmax, val_aupr, extra={"val_loss": float(val_loss)}, lambda_auprc=1.0, use_pareto=True)

                if stopper.step(val_loss, val_fmax, model):
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # test_fmax, test_aupr, test_loss = self.evaluate(self.dataset.test_loader, mode="Test")
            # print(f"Test - Fmax: {test_fmax:.4f}, AUPR: {test_aupr:.4f}, Loss: {test_loss:.4f}")

            # 最后一次模型也可评估
            last_test = self.evaluate_and_save_for_ckpt(tag="last", ckpt_path=self.model_path, split="test")

            # best by Fmax
            if getattr(self, "best_fmax_path", None) is not None:
                self.evaluate_and_save_for_ckpt(tag="bestFmax", ckpt_path=self.best_fmax_path, split="test")

            # best by AUPRC
            if getattr(self, "best_auprc_path", None) is not None:
                self.evaluate_and_save_for_ckpt(tag="bestAUPRC", ckpt_path=self.best_auprc_path, split="test")

            # bestBoth (score-based representative)
            if getattr(self, "best_both_path", None) is not None:
                self.evaluate_and_save_for_ckpt(tag="bestBoth", ckpt_path=self.best_both_path, split="test")

            # (optional) strict both improved
            if getattr(self, "best_both_strict_path", None) is not None and self.best_both_strict_path.exists():
                self.evaluate_and_save_for_ckpt(tag="bestBothStrict", ckpt_path=self.best_both_strict_path, split="test")


        def evaluate(self, dataloader, mode: str = "Valid"):
            """
            根据 dataloader + 当前 split，对整套数据计算:
            - 平均 loss
            - micro Fmax
            - micro AUPR
            """
            mode_lower = mode.lower()
            if "train" in mode_lower:
                base_g = self.dataset.train_g
            elif "val" in mode_lower:
                base_g = self.dataset.valid_g
            else:
                base_g = self.dataset.test_g

            self.model.eval()
            all_preds = []
            all_ema_preds = []
            all_labels = []
            total_loss = 0.0
            ema_total_loss = 0.0
            n_batches = 0

            amp = self.device.startswith("cuda") and torch.cuda.is_available()
            criterion = self.loss_fn

            with torch.no_grad():
                for input_nodes, output_nodes, blocks in dataloader:
                    # 只关心有 protein 输出的 batch
                    if "protein" not in output_nodes:
                        continue

                    # 1) 构造 batch 子图
                    sg = dgl.node_subgraph(base_g, input_nodes)

                    # 2) MSA / ProFun-SOM 特征
                    h_msa_dict = sg.ndata["h"]

                    # 3) ESM2 特征
                    esm2_dict = {}
                    if self.esm2_protein_features is not None and sg.num_nodes("protein") > 0:
                        p_parent = sg.nodes["protein"].data["parent_nid"]
                        esm2_dict["protein"] = self.esm2_protein_features[p_parent]

                    if self.esm2_go_features is not None and sg.num_nodes("go_annotation") > 0:
                        g_parent = sg.nodes["go_annotation"].data["parent_nid"]
                        esm2_dict["go_annotation"] = self.esm2_go_features[g_parent]

                    # 4) 前向：得到子图所有 protein 的 logits
                    if amp:
                        with autocast(device_type=self.device.split(":")[0], enabled=True):
                            out = self.model(sg, h_msa_dict, esm2_dict)
                            if self.ema_model is not None:
                                out_ema = self.ema_model.module(sg, h_msa_dict, esm2_dict) 
                            else:
                                out_ema = None
                    else:
                        out = self.model(sg, h_msa_dict, esm2_dict)
                        if self.ema_model is not None:
                            out_ema = self.ema_model.module(sg, h_msa_dict, esm2_dict)
                        else:
                            out_ema = None

                    logits_all = out["protein"]  # (N_protein_in_sg, num_classes)
                    ema_logits_all = out_ema["protein"] if out_ema is not None else None

                    # 5) 将 output_nodes['protein'] (base_g id) 映射到 sg 的 local index
                    out_p = output_nodes["protein"]    # in base_g id space
                    sg_base_ids = sg.nodes["protein"].data[dgl.NID].detach().cpu().tolist()
                    id2idx = {int(pid): i for i, pid in enumerate(sg_base_ids)}

                    try:
                        out_p_list = out_p.detach().cpu().tolist()
                        idx_list = [id2idx[int(pid)] for pid in out_p_list]
                    except KeyError:
                        # 极少出现不一致时，跳过这个 batch
                        continue

                    idx_tensor = torch.tensor(idx_list, device=self.device, dtype=torch.long)
                    logits = logits_all[idx_tensor]  # (B, num_classes)
                    ema_logits = ema_logits_all[idx_tensor] if ema_logits_all is not None else None

                    # 6) labels：通过 base_g.parent_nid -> full graph id -> get_label
                    parent_ids = base_g.nodes["protein"].data["parent_nid"][out_p]
                    parent_ids_list = parent_ids.detach().cpu().tolist()
                    # labels_np = self.dataset.get_label(parent_ids_list)
                    labels_np = self.dataset.get_label_eval(parent_ids_list)
                    labels = torch.tensor(labels_np, dtype=torch.float32, device=self.device)

                    if logits.ndim != 2 or logits.shape != labels.shape:
                        raise RuntimeError(
                            f"[eval shape mismatch] logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
                        )

                    loss = criterion(logits, labels)
                    ema_loss = criterion(ema_logits, labels) if ema_logits is not None else None
                    total_loss += float(loss.item())
                    ema_total_loss += float(ema_loss.item()) if ema_loss is not None else 0.0
                    n_batches += 1

                    probs = torch.sigmoid(logits)
                    all_preds.append(probs.detach().cpu().numpy())
                    if ema_logits is not None:
                        ema_probs = torch.sigmoid(ema_logits)
                        all_ema_preds.append(ema_probs.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())

            if n_batches == 0:
                return 0.0, 0.0, 0.0

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            all_ema_preds = np.concatenate(all_ema_preds, axis=0) if len(all_ema_preds) > 0 else None

            avg_loss = total_loss / n_batches
            ema_avg_loss = ema_total_loss / n_batches if self.ema_model is not None else None
            fmax, aupr = fmax_score(all_labels, all_preds, auprc=True)
            if all_ema_preds is not None:
                ema_fmax, ema_aupr = fmax_score(all_labels, all_ema_preds, auprc=True)
            else:
                ema_fmax, ema_aupr = None, None

            if ema_fmax is not None:
                print(f"Fmax: {fmax:.4f}, AUPR: {aupr:.4f}, Loss: {avg_loss:.4f}\n"
                      f"EMA Fmax: {ema_fmax:.4f}, EMA AUPR: {ema_aupr:.4f}, EMA Loss: {ema_avg_loss:.4f}")
                fmax = max(fmax, ema_fmax)
                aupr = max(aupr, ema_aupr)
                avg_loss = min(avg_loss, ema_avg_loss)
            else:
                print(f"Fmax: {fmax:.4f}, AUPR: {aupr:.4f}, Loss: {avg_loss:.4f}")
            return fmax, aupr, avg_loss

        # def compute_metrics(self, preds, labels):
        #     """计算 Fmax 和 AUPR"""
        #     # 过滤无标签的蛋白质和 GO 术语
        #     valid_proteins = np.where(labels.sum(axis=1) > 0)[0]
        #     valid_go_terms = np.where(labels.sum(axis=0) > 0)[0]

        #     if len(valid_proteins) == 0 or len(valid_go_terms) == 0:
        #         return 0.0, 0.0

        #     preds_filtered = preds[valid_proteins][:, valid_go_terms]
        #     labels_filtered = labels[valid_proteins][:, valid_go_terms]

        #     return f1_max(labels_filtered, preds_filtered, auprc=True)

        def save_best_by_metrics(self, epoch: int, val_fmax: float, val_auprc: float, extra: dict | None = None):
            """
            Save two best checkpoints separately:
            1) best by Fmax on validation
            2) best by AUPRC on validation

            Also writes a small json meta next to each ckpt for bookkeeping.
            """
            # ensure dir
            self.best_fmax_path.parent.mkdir(parents=True, exist_ok=True)

            def _dump_meta(meta_path: Path, payload: dict):
                meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

            # common payload
            base_meta = {
                "epoch": int(epoch),
                "val_fmax": float(val_fmax),
                "val_auprc": float(val_auprc),
            }
            if extra:
                base_meta.update(extra)

            # ---- best Fmax ----
            if val_fmax > self.best_fmax:
                self.best_fmax = float(val_fmax)
                self.best_fmax_epoch = int(epoch)

                ckpt = {
                    "epoch": int(epoch),
                    "model_state_dict": self.model.state_dict(),
                    "ema_model": self.ema_model.module.state_dict() if self.ema_model is not None else None,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
                    "best_fmax": self.best_fmax,
                    "best_auprc": self.best_auprc,
                }
                torch.save(ckpt, self.best_fmax_path)

                meta_path = self.best_fmax_path.with_suffix(".json")
                _dump_meta(meta_path, {**base_meta, "best_type": "Fmax"})
                print(f"[CKPT] Saved best-Fmax checkpoint @ epoch={epoch} to {self.best_fmax_path}")

            # ---- best AUPRC ----
            if val_auprc > self.best_auprc:
                self.best_auprc = float(val_auprc)
                self.best_auprc_epoch = int(epoch)

                ckpt = {
                    "epoch": int(epoch),
                    "model_state_dict": self.model.state_dict(),
                    "ema_model": self.ema_model.module.state_dict() if self.ema_model is not None else None,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
                    "best_fmax": self.best_fmax,
                    "best_auprc": self.best_auprc,
                }
                torch.save(ckpt, self.best_auprc_path)

                meta_path = self.best_auprc_path.with_suffix(".json")
                _dump_meta(meta_path, {**base_meta, "best_type": "AUPRC"})
                print(f"[CKPT] Saved best-AUPRC checkpoint @ epoch={epoch} to {self.best_auprc_path}")

        def load_checkpoint(self, ckpt_path: Path, strict: bool = True):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)  # 兼容只保存 state_dict 的情况
            self.model.load_state_dict(state, strict=strict)
            return ckpt

        def save_predictions(self, tag: str = "last", dataloader=None, split_name: str = "test"):
            """
            Save predictions for a given dataloader.
            - tag:      arbitrary tag, e.g. "best", "last"
            - split:    "train" / "valid" / "test"
            """

            # 1) 选择 dataloader 和 对应的 base graph
            if dataloader is None:
                if split_name == "train":
                    dataloader = self.dataset.train_loader
                    base_g = self.dataset.train_g
                elif split_name == "valid":
                    dataloader = self.dataset.valid_loader
                    base_g = self.dataset.valid_g
                else:
                    dataloader = self.dataset.test_loader
                    base_g = self.dataset.test_g
            else:
                # 如果外部传入 dataloader，则按 split_name 选择 base_g
                if split_name == "train":
                    base_g = self.dataset.train_g
                elif split_name == "valid":
                    base_g = self.dataset.valid_g
                else:
                    base_g = self.dataset.test_g

            self.model.eval()
            all_preds = []
            all_labels = []
            all_protein_names = []

            with torch.no_grad():
                for input_nodes, output_nodes, blocks in dataloader:
                    # 只关心有 protein 的 batch
                    if "protein" not in output_nodes:
                        continue

                    # 2) 构造 batch 子图（基于当前 split 的 base_g）
                    sg = dgl.node_subgraph(base_g, input_nodes)

                    # 3) MSA / ProFun-SOM 特征（子图里已经带有 ndata['h']）
                    h_msa_dict = sg.ndata["h"]    # hetero: dict-like per ntype

                    # 4) ESM2 特征：通过 parent_nid 从全量矩阵 gather
                    esm2_dict = {}
                    if self.esm2_protein_features is not None and sg.num_nodes("protein") > 0:
                        p_parent = sg.nodes["protein"].data["parent_nid"]   # full graph id
                        esm2_dict["protein"] = self.esm2_protein_features[p_parent]

                    if self.esm2_go_features is not None and sg.num_nodes("go_annotation") > 0:
                        g_parent = sg.nodes["go_annotation"].data["parent_nid"]
                        esm2_dict["go_annotation"] = self.esm2_go_features[g_parent]

                    # 5) 前向：模型返回子图内所有 protein 的 logits
                    out = self.model(sg, h_msa_dict, esm2_dict)
                    logits_all = out["protein"]   # (N_protein_in_sg, num_classes)

                    # 6) 将 output_nodes['protein']（base_g id）映射到 sg 的 local index
                    out_p = output_nodes["protein"]   # in base_g id space
                    # sg 中每个 protein 节点对应的 base_g id，保存在 dgl.NID
                    sg_base_ids = sg.nodes["protein"].data[dgl.NID].detach().cpu().tolist()
                    id2idx = {int(pid): i for i, pid in enumerate(sg_base_ids)}

                    try:
                        out_p_list = out_p.detach().cpu().tolist()
                        idx_list = [id2idx[int(pid)] for pid in out_p_list]
                    except KeyError:
                        # 保险起见，如果有找不到的，跳过这个 batch
                        continue

                    idx_tensor = torch.tensor(idx_list, device=self.device, dtype=torch.long)
                    preds_batch = torch.sigmoid(logits_all[idx_tensor])   # (B, num_classes)

                    # 7) labels：通过 base_g 的 parent_nid -> full graph id -> get_label
                    parent_ids = base_g.nodes["protein"].data["parent_nid"][out_p]
                    parent_ids_list = parent_ids.detach().cpu().tolist()
                    labels_list = self.dataset.get_label(parent_ids_list)
                    labels_batch = torch.tensor(labels_list, dtype=torch.float32, device=self.device)

                    # 8) protein names：推荐用 full graph parent id 做 key
                    protein_names_batch = []
                    for pid in parent_ids_list:
                        if hasattr(self, "protein_name_dict") and isinstance(self.protein_name_dict, dict):
                            protein_names_batch.append(
                                self.protein_name_dict.get(pid, f"protein_{pid}")
                            )
                        else:
                            # 如果没有字典，就用占位名
                            protein_names_batch.append(f"protein_{pid}")

                    # 9) 累积结果
                    all_preds.append(preds_batch.detach().cpu().numpy())
                    all_labels.append(labels_batch.detach().cpu().numpy())
                    all_protein_names.extend(protein_names_batch)

            # 10) 拼接 & 保存
            if len(all_preds) > 0:
                all_preds = np.concatenate(all_preds, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
            else:
                all_preds = np.zeros((0, self.dataset.go_num), dtype=np.float32)
                all_labels = np.zeros((0, self.dataset.go_num), dtype=np.float32)

            result_path = (
                self.model_path.parent
                / f"inductive_HGAT_ESM2_{self.dataset.go_name}_v3_predictions_{split_name}_{tag}.npz"
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                result_path,
                protein_names=np.array(all_protein_names),
                predictions=all_preds,
                labels=all_labels,
                go_terms=np.array(self.go_terms_list),
            )
            print(f"[PRED] Saved {split_name} predictions ({tag}) -> {result_path}")


        def evaluate_and_save_for_ckpt(self, tag: str, ckpt_path, split: str = "test"):
            """
            Load a checkpoint, evaluate on split loader, and save predictions with tag.

            Args:
            tag: name suffix in output files, e.g., "bestFmax", "bestAUPRC", "bestBoth"
            ckpt_path: Path or str to checkpoint .pt
            split: "test" | "valid" | "train" (default "test")
            """
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                print(f"[WARN] ckpt not found: {ckpt_path}")
                return None

            # 1) load checkpoint
            _ = self.load_checkpoint(ckpt_path, strict=True)
            print(f"[CKPT] Loaded: {ckpt_path}")

            # 2) choose loader
            if split == "test":
                loader = self.dataset.test_loader
            elif split == "valid":
                loader = self.dataset.valid_loader
            elif split == "train":
                loader = self.dataset.train_loader
            else:
                raise ValueError(f"Unknown split={split}. Use test/valid/train.")

            # 3) evaluate
            fmax, aupr, loss = self.evaluate(loader, mode=split.capitalize())
            print(f"[{tag}] {split.capitalize()} - Fmax: {fmax:.4f}, AUPR: {aupr:.4f}, Loss: {loss:.4f}")

            # 4) save preds
            self.save_predictions(tag=tag, dataloader=loader, split_name=split)

            return {"split": split, "tag": tag, "fmax": fmax, "aupr": aupr, "loss": loss, "ckpt": str(ckpt_path)}


        @staticmethod
        def _dominates(a_f: float, a_p: float, b_f: float, b_p: float, eps: float = 1e-12) -> bool:
            """
            return True if (a_f, a_p) dominates (b_f, b_p):
            a_f >= b_f and a_p >= b_p and at least one strictly greater.
            """
            ge_f = a_f + eps >= b_f
            ge_p = a_p + eps >= b_p
            gt_any = (a_f > b_f + eps) or (a_p > b_p + eps)
            return ge_f and ge_p and gt_any

        def _save_ckpt_with_meta(self, ckpt_path: Path, meta: dict):
            ckpt = {
                "epoch": int(meta.get("epoch", -1)),
                "model_state_dict": self.model.state_dict(),
                "ema_model": self.ema_model.module.state_dict() if self.ema_model is not None else None,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
                "meta": meta,
            }
            torch.save(ckpt, ckpt_path)
            ckpt_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

        def save_best_both(self, epoch: int, val_fmax: float, val_auprc: float, extra: dict | None = None,
                        lambda_auprc: float = 1.0, use_pareto: bool = True):
            """
            Save "bestBoth" which targets being good on BOTH metrics.

            Two mechanisms:
            1) strict-both: only save if BOTH metrics improve over previous strict-both best.
            2) pareto+score (recommended): maintain Pareto front; pick representative by score:
                    score = val_fmax + lambda_auprc * val_auprc
                If score improves, save as bestBoth.

            Args:
            lambda_auprc: weight for AUPRC in scalar score. Start with 1.0; you can tune.
            use_pareto: whether to maintain Pareto front and select representative.
            """
            meta = {
                "epoch": int(epoch),
                "val_fmax": float(val_fmax),
                "val_auprc": float(val_auprc),
            }
            if extra:
                meta.update(extra)

            # ---- (A) strict both improved ----
            if (val_fmax > self.best_both_strict_fmax) and (val_auprc > self.best_both_strict_auprc):
                self.best_both_strict_fmax = float(val_fmax)
                self.best_both_strict_auprc = float(val_auprc)
                self.best_both_strict_epoch = int(epoch)
                meta_strict = {**meta, "best_type": "BothStrict"}
                self._save_ckpt_with_meta(self.best_both_strict_path, meta_strict)
                print(f"[CKPT] Saved best-BothStrict @ epoch={epoch} to {self.best_both_strict_path}")

            # ---- (B) pareto front maintenance ----
            if use_pareto:
                # check if new point is dominated by existing pareto points
                dominated = False
                for p in self.pareto_front:
                    if self._dominates(p["fmax"], p["auprc"], val_fmax, val_auprc):
                        dominated = True
                        break

                if not dominated:
                    # remove points dominated by new point
                    new_front = []
                    for p in self.pareto_front:
                        if not self._dominates(val_fmax, val_auprc, p["fmax"], p["auprc"]):
                            new_front.append(p)
                    self.pareto_front = new_front

                    # save this pareto candidate ckpt
                    pareto_ckpt = self.pareto_dir / f"epoch{epoch:03d}_F{val_fmax:.4f}_P{val_auprc:.4f}.pt"
                    meta_pareto = {**meta, "best_type": "ParetoCandidate"}
                    self._save_ckpt_with_meta(pareto_ckpt, meta_pareto)
                    self.pareto_front.append({
                        "epoch": int(epoch),
                        "fmax": float(val_fmax),
                        "auprc": float(val_auprc),
                        "path": str(pareto_ckpt),
                    })

                    # persist pareto front index
                    (self.pareto_dir / "pareto_front.json").write_text(
                        json.dumps(self.pareto_front, indent=2, ensure_ascii=False)
                    )
                    print(f"[CKPT] Added Pareto candidate @ epoch={epoch} (front size={len(self.pareto_front)})")

            # ---- representative "bestBoth" by scalar score ----
            score = float(val_fmax + lambda_auprc * val_auprc)
            if score > self.best_both_score:
                self.best_both_score = score
                self.best_both_epoch = int(epoch)
                meta_both = {**meta, "best_type": "BestBothScore", "lambda_auprc": float(lambda_auprc), "score": score}
                self._save_ckpt_with_meta(self.best_both_path, meta_both)
                print(f"[CKPT] Saved best-BothScore @ epoch={epoch} score={score:.6f} to {self.best_both_path}")


    # 你的 HGAT_ESM2_V3_Trainer 类定义也建议放在 main() 内部或单独模块里
    trainer = HGAT_ESM2_V3_Trainer(trainer_args)
    trainer.train()

if __name__ == '__main__':
    main()


