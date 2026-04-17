import sys
import pathlib as P
sys.path.append('..')
prj_root = str(P.Path(__file__).parent.parent)
if prj_root not in sys.path:
    sys.path.append(prj_root)
from util.utils import EarlyStopping
from models.HGAT import *
from dataset.transductive_learning_dataset import *
from time import time
import torch
torch.cuda.empty_cache()
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import dgl
from dgl.sampling import RandomWalkNeighborSampler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, f1_score, auc
import pathlib as P
import concurrent.futures

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

warnings.filterwarnings("ignore", category=UserWarning)




class HGAT_Trainer(object):

    def __init__(self, args):
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.device = args['device']
        self.dataset = DBLPDataset('data', args['go_name'], 'GCN', args['batch_size'], self.device, result_label=args['result_label'])
        self.g = self.dataset.g.to(self.device)
        self.model = HGAT(self.dataset.node_type, num_classes=self.dataset.go_num, feature_dim=self.dataset.feature_dim, hidden_dim=128, num_layers=2)
        self.model = self.model.to(self.device)
        self.model_path = args['model_path'] +'/' + args['model_name']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.loss_func = nn.BCELoss()
        self.best_fmax = 0
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.scheduler = OneCycleLR(self.optimizer,max_lr=self.lr, total_steps=self.epoch*len(self.dataset.train_loader))
        # self.label = self.dataset.get_label()
        # self.input_feature = self.dataset.input_feature.to(self.device)

        self.category = 'protein'

        self.meta_paths_dict = {
                'mp_0': [('protein', 'interacts_0', 'go_annotation')],
                'mp_1': [('protein', 'interacts_1', 'protein')],
                'mp_2': [('go_annotation', 'interacts_2', 'go_annotation')]
            }

    def train(self):
        # train_dataset = DBLPDataset(features_list, self.device)
        # dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        model = self.model

        for epoch in range(self.epoch):
            train_loss = self.train_step(self)
            
            modes = ['valid']
            fmax, aupr, val_loss = self.test_step(modes=modes)
            print("Epoch: {}, Train loss: {}, Valid loss: {} ".format(epoch, train_loss, val_loss)
                                       + "fmax: {}, aupr: {}".format(fmax, aupr))
        # self.load_model()
        modes = ['test']
        fmax, aupr, loss = self.test_step(modes=modes)
        # for key in metric_dict:
        #     print(metric_dict[key] + losses[key].cpu())
        return fmax, aupr
            
                
    def train_step(self, logits=None):
            # return self._full_train_step()
            self.model.train()
            scaler = GradScaler()
            loss_all = 0.0
            loader_tqdm = tqdm(self.dataset.train_loader, ncols=120)
            for i, (sample_nodes, seed, blocks) in enumerate(loader_tqdm):
            # for i, (seed, subgraph_protein_nodes, subgraph) in enumerate(loader_tqdm): 
                subgraph = dgl.node_subgraph(self.dataset.g, sample_nodes)
                # subgraph = self.dataset.blocks_to_graph(blocks)
                # subgraph = blocks
                # subgraph = blocks_to_hetero_graph(blocks)
                # h_dict = {}
                # mp_name_input_nodes_dict = sample_nodes[self.category]

                # for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                #     h_dict[meta_path_name] = self.input_feature.forward_nodes({self.category: input_nodes.to(self.device)})

                # label = self.label[seed[self.category]].to(self.device)
                if 'protein' not in seed.keys():
                    break
                seed = seed['protein']
                seed_ = self.dataset.g.ndata[dgl.NID]['protein'][seed]
                label = self.dataset.get_label(seed_.cpu().numpy().tolist())
                h_dict = subgraph.ndata['h']
                
                pred = self.model(subgraph, h_dict)

                # seed_indices = [torch.where(subgraph_protein_nodes == x)[0].item() for x in seed]
                
                seed_indices = [torch.where(sample_nodes['protein'] == x)[0].item() for x in seed]
                pred = pred['protein'][seed_indices]
                
                # label = th.argmax(th.tensor(label), dim=1).to(self.device)
                label = torch.tensor(label).to(self.device)
                # train_loss = F.nll_loss(F.log_softmax(pred, dim=1), label.long())
                train_loss = self.loss_fn(pred, label.float())
                loss_all += train_loss
                scaler.scale(train_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                
                self.scheduler.step()


                # self.optimizer.zero_grad()
                # train_loss.backward()
                # self.optimizer.step()
                # accumulation_steps = 4  # 每个大批次分成的小批次数量
                # train_loss.backward()
                # self.optimizer.step()
                # self.optimizer.zero_grad()
                # 每 accumulation_steps 个小批次进行一次优化更新
                # if (i + 1) % accumulation_steps == 0:
                #     self.optimizer.step()
            
            return loss_all / (i + 1)

    def test_step(self, modes):
        self.model.eval()

        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.dataset.train_loader, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.dataset.valid_loader, ncols=120)
                elif mode == 'test':
                    self.load_model()
                    loader_tqdm = tqdm(self.dataset.test_loader, ncols=120)
                y_trues = []
                y_predicts = []

                for i, (sample_nodes, seed, blocks) in enumerate(loader_tqdm):
                # for i, (seed, subgraph_protein_nodes, subgraph) in enumerate(loader_tqdm):
                    with autocast():
                        seed = seed['protein']
                        subgraph = dgl.node_subgraph(self.g, sample_nodes)
                        # subgraph = self.dataset.blocks_to_graph(blocks)
                        # subgraph = blocks_to_hetero_graph(blocks)
                        label = self.dataset.get_label(seed.cpu().numpy().tolist())
                        h_dict = subgraph.ndata['h']
                        pred = self.model(subgraph, h_dict)

                        # seed_indices = [torch.where(subgraph_protein_nodes == x)[0].item() for x in seed]
                        seed_indices = [torch.where(sample_nodes['protein'] == x)[0].item() for x in seed]
                        pred = pred['protein'][seed_indices]
                        # label = th.argmax(th.tensor(label), dim=1).to(self.device)
                        label = torch.tensor(label).to(self.device)
                        # loss = F.nll_loss(pred, label.long())
                        
                        if torch.isnan(pred).any():
                            print(pred)
                            raise ValueError("nan")
                        
                        loss = self.loss_fn(pred, label.float())
                        loss_all += loss
                        pred = torch.nn.functional.sigmoid(pred)
                        y_trues.append(label.to(torch.float32).detach().cpu())
                        y_predicts.append(pred.to(torch.float32).detach().cpu())    
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                # evaluator = self.task.get_evaluator(name='f1')
                # metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                # fmax, aupr = self.calculate_metrics(y_trues, y_predicts)
                fmax, aupr = self.get_fmax_aupr(y_trues, y_predicts)
                print('fmax:{}, aupr:{}'.format(fmax, aupr))
                if mode == 'valid':
                    if fmax > self.best_fmax:
                        self.best_fmax = fmax
                        self.save_model()
                elif mode == 'test':
                    # index = self.dataset.test_idx.unsqueeze(1)
                    # results = torch.cat((index.detach().cpu(), y_predicts), dim=1).numpy()
                    # np.savetxt(self.dataset.result_path, results)
                    self.dataset.save_results(y_predicts, y_trues)
                # metric_dict[mode] = self.calculate_metrics(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                metric_dict[mode] = (fmax, aupr)
                loss_dict[mode] = loss_all
        # return metric_dict, loss_dict
        return fmax, aupr, loss_all/(i+1)
                
                # h_dict = self.input_feature
                # h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
                # pred = self.model(self.g, h_dict)
                # masks = {}
                # if mode == "valid":
                #     pred = pred[self.valid_idx]
                # elif mode == "test":
                #     pred = pred[self.test_idx]
                
                # loss = self.loss_func(pred, label)
                # return loss


            # loss_all = 0.0
            # loader_tqdm = tqdm(self.train_loader, ncols=120)
            # for i, (sample_nodes, seed,g) in enumerate(loader_tqdm):
            #     # mmd_loss=0
            #     # l1_loss=0
            #     # cls_lossS_MP=0
            #     h_dict = {}
            #     mp_name_input_nodes_dict = sample_nodes[self.category]
            #     for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
            #         h_dict[meta_path_name] = self.model.input_featureS.forward_nodes({self.category: input_nodes})
            #     label = self.label[seed[self.category]].to(self.device)
            #     pred = self.model(g, h_dict={self.category:h_dict})

            #     loss = self.loss_func(pred, label)
            #     loss_all += loss
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            # return loss_all / (i + 1)

    def calculate_metrics(self, y_true, y_pred):
        num_classes = y_true.shape[1]
        fmax_list = []
        aupr_list = []
        
        for i in range(num_classes):
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            
            aupr = auc(recall, precision)
            aupr_list.append(aupr)
            
            f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
            
            fmax = np.nanmax(f1_scores)
            fmax_list.append(fmax)
        
        final_fmax = np.mean(fmax_list)
        final_aupr = np.mean(aupr_list)
        
        return final_fmax, final_aupr

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        
    def get_fmax_aupr(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        fmax = np.max(f1_scores)
        aupr = auc(recall, precision)
        return fmax, aupr
    
def get_nodes_dict(hg):
    n_dict = {}
    for n in hg.ntypes:
        n_dict[n] = hg.num_nodes(n)
    return n_dict

def blocks_to_hetero_graph(blocks):
    graph_list = []
    
    node_types = ['protein', 'go_annotation']
    edge_types = ['interacts_0', 'interacts_1', 'interacts_2', '_interacts_0', '_interacts_1', '_interacts_2']
    edge_types_full = {'interacts_0':('protein', 'interacts_0', 'go_annotation'),
                        '_interacts_0':('go_annotation', '_interacts_0', 'protein'),
                        'interacts_1':('protein', 'interacts_1', 'protein'),
                        '_interacts_1':('protein', '_interacts_1', 'protein'),
                        'interacts_2':('go_annotation', 'interacts_2', 'go_annotation'),
                        '_interacts_2':('go_annotation', '_interacts_2', 'go_annotation')}

    for block in blocks:
        graph_data = {}
        node_data = {}
        for ntype in node_types:
            node_data[ntype] = block.nodes[ntype].data['h']
        for etype in edge_types:
            # node_l = block.edges(etype=etype)[0]
            # node_r = block.edges(etype=etype)[1]
            
            graph_data[edge_types_full[etype]] = block.edges(etype=etype)
        g = dgl.heterograph(graph_data)
        for ntype in node_types:
            g.nodes[ntype].data['h'] = node_data[ntype]
        graph_list.append(g)
    
    return graph_list

import yaml
import click

@click.command()
@click.option('-d', '--device', default = 'cpu')
@click.option('-go', '--go_name')
@click.option('-e', '--epoch', default = 10)
@click.option('-b', '--batch_size', default=16)
@click.option('-l', '--lr', default = 5e-4)
@click.option('--weight_decay', default = 5e-4)
@click.option('--model_path', default = prj_root + '/models/')
@click.option('--model_name')
@click.option('--result_label')
def main(device, go_name, epoch, batch_size, lr, model_path, model_name, result_label, weight_decay):
    results = {}
    args = {'device': torch.device(device),
        'go_name': go_name,
        # 'model_path':prj_root + '/models/hgat_esm1_mmseqs_' + name + '_best',
        'model_path': model_path,
        'model_name': model_name,
        'result_label':result_label,
        'epoch':epoch,
        'batch_size':batch_size,
        'lr':lr,
        'weight_decay':weight_decay}
        
    trainer = HGAT_Trainer(args=args)
    fmax, aupr = trainer.train()
    results[go_name] = 'fmax:{}, aupr:{}'.format(fmax, aupr)
    print(results)

if __name__ == '__main__':
    main()