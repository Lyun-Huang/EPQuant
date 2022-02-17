#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os.path as osp
import os
import gc
import torch
import torch.nn.functional as F
import sys
import argparse
from tqdm import tqdm
import torch_geometric
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from quantization.sq.utils import quant_framework
from quantization.epq.pqact import ActPQ
from quantization.utils import QParam, Layer_Qparam, sizeTracker, result_container, fetchAssign
import logging

logger = logging.getLogger("")
logger.setLevel(logging.INFO)  #DEBUG < INFO < WARNING < ERROR < CRITICAL

# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6'

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    parser.add_argument('--dataset', type=str, default='Reddit')
    parser.add_argument('--model', type=str, default='GS-mean')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--nruns', type=int, default=1, help='number of independent runs')
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--ncents', type=int, default=256, 
                        help='the upper limit of learned clusters and the upper limit of learned clusters for each batch if mini_batch is true')
    parser.add_argument('--try_cluster', type=int, default=15,
                        help='number of attempts to find more centroids')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='number of iteration for cluster')
    parser.add_argument('--mini_batch', action="store_true", default=True, 
                        help='apply batch method')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of nodes in each batch')
    parser.add_argument('--path', type=str, default=f'./pq_data/reddit/')
    parser.add_argument('--pqnt', action="store_true", 
                        help='apply EPQ on input data')
    parser.add_argument('--act_qnt', action="store_true", 
                        help='apply SQ on input data')
    parser.add_argument('--wt_qnt', action="store_true", 
                        help='apply SQ on weight')
    parser.add_argument('--bits', type=tuple, default=(8,8),
                        help='quantization bits of each layer')
    parser.add_argument('--wf', action="store_true", 
                        help='write result to file')
    parser.add_argument('--f', type=str, default='result.txt', help='path of result file')
#    parser.add_argument('--verbose', type=str, default='INFO')
    
    args = parser.parse_args()
    print(args)
    return args


def dataProcess(dataset, use_gdc=False):
    if dataset == 'Reddit':
        from torch_geometric.datasets import Reddit
        path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', 'Reddit')
        dataset = Reddit(path)
    elif dataset == 'Amazon2M':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name = "ogbn-products", root = './data/') 
    else:
        pass
    
    return dataset

args = argParse()
dataset = dataProcess(args.dataset)
data = dataset[0]

if args.dataset == 'Reddit':
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], 
                                   batch_size=1024, shuffle=True,
                                   num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)
elif args.dataset == 'Amazon2M':  
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    
    train_mask = val_mask = test_mask = torch.Tensor([False]*data.num_nodes).bool()
    train_mask[train_idx]=True
    val_mask[valid_idx]=True
    test_mask[test_idx]=True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], 
                                   batch_size=1024, shuffle=True,
                                   num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.num_layers = 2
        self.convs0 = SAGEConv(in_channels, hidden_channels)
        self.convs1 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x_i):
        x, adjs = x_i
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = getattr(self, f"convs{i}")((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = getattr(self, f"convs{i}")((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

from quantization.utils import avaliable_GPU
device_id = avaliable_GPU()
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.x.size()[1], 256, dataset.num_classes)
# model = torch_geometric.nn.DataParallel(model.cuda())
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

save = False
load = not save
wt_qnt = args.wt_qnt
act_qnt = args.act_qnt
pqnt = args.pqnt

layer0_input_qparam = QParam(
     n_centroids=args.ncents, 
     block_size=args.block_size,
     batch_size=args.batch_size,
     mini_batch=args.mini_batch,
     save=save,
     load=load,
     # path=osp.join(args.path, args.dataset.lower()),
     path=args.path,
)

layer0_act_qparam = QParam(
     sqnt=act_qnt,
     bits=args.bits,
     p=0.3, 
)

layer0_wt_qparam = QParam(
     sqnt=wt_qnt,
     bits=args.bits,
     p=0.3,
)

layer0_qparam = Layer_Qparam(input_qparam=layer0_input_qparam, wt_qparam=layer0_wt_qparam, act_qparam=layer0_act_qparam)

layer1_input_qparam = QParam()

layer1_act_qparam = QParam(
     sqnt=act_qnt,
     bits=args.bits,
     p=0.3, 
)

layer1_wt_qparam = QParam(
     sqnt=wt_qnt,
     bits=args.bits,
     p=0.3,
)

layer1_qparam = Layer_Qparam(input_qparam=layer1_input_qparam, wt_qparam=layer1_wt_qparam, act_qparam=layer1_act_qparam)

qnt_param = {'layer_0':layer0_qparam, 'layer_1':layer1_qparam}
quant_framework(model, **qnt_param)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch, x):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, edge_index in train_loader:
        adjs = [adj.to(device) for adj in edge_index]
        optimizer.zero_grad()
        out = model([x[n_id], adjs])
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test(x):
    model.eval()
    out = model.inference(x)
    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


if pqnt and load:
    pact = ActPQ(
        model, 
        module_name="SAGE", 
        n_centroids=layer0_input_qparam.n_centroids, 
        block_size=layer0_input_qparam.block_size, 
        try_cluster=layer0_input_qparam.try_cluster, 
        n_iter=layer0_input_qparam.n_iter,
        eps=layer0_input_qparam.eps,
        load=True,
        path=layer0_input_qparam.path,
        mini_batch=layer0_input_qparam.mini_batch,
        batch_size=layer0_input_qparam.batch_size,
    )
    x = pact.input_quant(x).to(device)
    
if load:
    best_test_acc=0
    for epoch in range(1, args.epoch+1):
        loss, acc = train(epoch, x)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test(x)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        

