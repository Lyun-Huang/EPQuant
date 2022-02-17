#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os.path as osp
import sys
import argparse
from quantization.sq.utils import quant_framework
import logging
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from quantization.utils import QParam, Layer_Qparam, sizeTracker, result_container, fetchAssign
from benchmark import GCN, GAT
from numpy import mean
import time

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gdc', action='store_true',
                        help='Use GDC preprocessing.')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--nruns', type=int, default=1, help='number of independent runs')
    parser.add_argument('--block_size', type=int, default=22)
    parser.add_argument('--ncents', type=int, default=64, 
                        help='the upper limit of learned clusters and the upper limit of learned clusters for each batch if mini_batch is true')
    parser.add_argument('--try_cluster', type=int, default=15,
                        help='number of attempts to find more centroids')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='number of iteration for cluster')
    parser.add_argument('--mini_batch', action="store_true", 
                        help='apply batch method')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of nodes in each batch')
    parser.add_argument('--path', type=str, default=f'./pq_data/')
    parser.add_argument('--pqnt', action="store_true", #default=True,
                        help='apply EPQ on input data')
    parser.add_argument('--act_qnt', action="store_true", #default=True,
                        help='apply SQ on input data')
    parser.add_argument('--wt_qnt', action="store_true", #default=True,
                        help='apply SQ on weight')
    parser.add_argument('--bits', type=tuple, default=(8,8),
                        help='quantization bits of each layer')
    parser.add_argument('--wf', action="store_true", #default=True, 
                        help='write result to file')
    parser.add_argument('--pretrained', action="store_true", #default=True,
                        help='use pretrained model')
    parser.add_argument('--inf_time', action="store_true", #default=True,
                        help='record inference time')
    parser.add_argument('--print_result', action="store_true", #default=True,
                        help='')
    parser.add_argument('--fast', action="store_true", #default=True,
                        help='no need to download datasets, use data already quantized by EPQ.')
    parser.add_argument('--f', type=str, default='result.txt', help='path of result file')
#    parser.add_argument('--verbose', type=str, default='INFO')
    
    args = parser.parse_args()
    print(args)
    return args

def dataProcess(dataset, fast=False, use_gdc=False):
    if fast:
        data = torch.load(f'./pq_data/{dataset.lower()}/data.pth')
        dataset = None
    else:
        if dataset == 'Reddit':
            from torch_geometric.datasets import Reddit
            path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', 'Reddit')
            dataset = Reddit(path)
        elif dataset == 'Amazon2M':
            from ogb.nodeproppred import PygNodePropPredDataset
            dataset = PygNodePropPredDataset(name = "ogbn-products", root = './data/') 
        else:
            from torch_geometric.datasets import Planetoid
            path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', dataset)
            dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        
        data = dataset[0]

    if use_gdc:
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=128,
                                               dim=0), exact=True)
        data = gdc(data)

    return data, dataset

def train(data,model,optimizer):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test(data,model, time_record=False):
    if time_record:
        start_time = time.time()

    model.eval()
    logits, accs = model(data), []
    if time_record:
        end_time = time.time()
        elaps_time = end_time - start_time
        print(f'Time for inference is {elaps_time}\n')
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

if __name__ == "__main__":
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)  #DEBUG < INFO < WARNING < ERROR < CRITICAL
    args = argParse()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    block_size = 4
    n_cents = 64
    data, dataset = dataProcess(args.dataset, args.fast, args.use_gdc)
    num_classes = data.num_classes if args.fast else dataset.num_classes
    data_size = data.size if args.fast else data.x.size()
    if args.fast:
        data.x = torch.empty(data_size) 
    data = data.to(device)
    
    test_acc_list = []
    test_lo = test_hi = test_mean = 0
    for i in range(args.nruns):  
        if args.model == 'GCN':
            model = GCN(data_size[1], args.hidden, num_classes).to(device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=args.wd),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=args.lr)  # Only perform weight-decay on first convolution.
        elif args.model == 'GAT':
            model = GAT(data_size[1], num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            logger.error(f'{model} is not support yet.')
            sys.exit(0)
        if args.pretrained:
            model.load_state_dict(torch.load('./parameter.pkl'))
        
        save = bool(0)
        load = not save
        wt_qnt = args.wt_qnt
        act_qnt = args.act_qnt
        pqnt = args.fast or args.pqnt
        
        layer0_input_qparam = QParam(
             pqnt=pqnt, 
             n_centroids=args.ncents, 
             block_size=args.block_size,
             batch_size=args.batch_size,
             mini_batch=args.mini_batch,
             save=save,
             load=load,
             path=osp.join(args.path, args.dataset.lower()),
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
        model = model.to(device)
        

        best_val_acc = test_acc = best_test_acc = 0
        if not args.pretrained:
            for epoch in range(1, args.epoch+1):
                train(data, model, optimizer)
                train_acc, val_acc, tmp_test_acc = test(data, model)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                if epoch % 50 == 0:
                    print(log.format(epoch, train_acc, best_val_acc, best_test_acc))
            test_acc_list.append(best_test_acc)
        else:
            train_acc, val_acc, tmp_test_acc = test(data, model, args.inf_time)
            log = 'Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(train_acc, val_acc, tmp_test_acc))
                            
            
                
    if args.print_result:
        if not args.pretrained:
            test_mean = mean(test_acc_list)
            test_lo = test_mean - min(test_acc_list)
            test_hi = max(test_acc_list) - test_mean
        else:
            test_mean = tmp_test_acc
            test_lo = test_hi = 0
    #        print(f'test result: mean: {test_mean:.4f}, lo: {test_lo:.4f}, hi: {test_hi:.4f}\n')
        if args.pqnt or args.fast:
            assignments, centroids = fetchAssign(layer0_input_qparam, data_size[0], device)
        else:
            assignments = torch.empty(1)
            
        if args.model == 'GCN':
            input_sizes = (data_size, (data_size[0], args.hidden)) 
            model_param = {'model':'GCN', 'n_feature':data_size[1], 'n_hid':args.hidden, 
                           'n_class':num_classes, 'mean':test_mean, 
                           'lo':test_lo, 'hi':test_hi, 
                           'CSR':True, 'assignments':assignments.cpu()}
        elif args.model == 'GAT':
            input_sizes = (data_size, (data_size[0], args.hidden*args.nheads))
            model_param = {'model':'GAT', 'n_feature':data_size[1], 
                           'n_hid':args.hidden, 'n_heads':args.nheads, 'n_class':num_classes, 
                           'in_channel_num':1, 'mean':test_mean, 'lo':test_lo, 'hi':test_hi,
                           'CSR':True, 'assignments':assignments.cpu()}
            
        cps_res, acc_res = result_container(qnt_param, *input_sizes, **model_param)
        
        if args.wf:
            f=open(args.f, 'w+')
            f.write(f'\nblocksize:{args.block_size}, dataset:{args.dataset}, model: {args.model}\n')
            f.write(cps_res.get_string())
            f.write(acc_res.get_string())
            f.close()
    
    if not args.pretrained:
        torch.save(model.state_dict(), 'parameter.pkl')
