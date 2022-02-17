#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import ZeroPad2d
import torch_scatter
import torch
import math
from prettytable import PrettyTable

class Qparam:
    
    def __init__(self, 
                 int_qnt=True,
                 bits=8,
                 noise=0, 
                 input_pq=False, 
                 n_centroids=256, 
                 block_size=4,
                 save=False,
                 load=False,
                 path='',
                 ):
        self.int_qnt = int_qnt
        self.bits = bits
        self.p = noise
        self.n_centroids = n_centroids
        self.block_size = block_size
        self.input_pq = input_pq
        self.save = save
        self.load = load
        self.path = path
        
    def update(self, 
             int_qnt=True,
             bits=8,
             noise=0, 
             input_pq=False, 
             n_centroids=256, 
             block_size=4,
             save=False,
             load=False,
             path='',
             ):
        self.int_qnt = int_qnt
        self.bits = bits
        self.p = noise
        self.n_centroids = n_centroids
        self.block_size = block_size
        self.input_pq = input_pq
        self.save = save
        self.load = load
        self.path = path

class QParam:
    
    def __init__(self, 
                 sqnt=False,
                 bits: tuple = (8,8),
                 p=0, 
                 clamp_threshold=5,
                 update_step=1000,
                 method='histogram',
                 pqnt=False, 
                 n_centroids=256, 
                 block_size=4,
                 n_iter=15,
                 eps=1e-6,
                 try_cluster=100,
                 batch_size=1,
                 mini_batch=False,
                 save=False,
                 load=False,
                 path='',
                 ):
        self.sqnt = sqnt
        self.bits = bits
        self.p = p
        self.clamp_threshold = clamp_threshold
        self.update_step = update_step
        self.method = method
        self.pqnt = pqnt
        self.n_centroids = n_centroids
        self.block_size = block_size
        self.n_iter = n_iter
        self.eps = eps
        self.try_cluster = try_cluster
        self.batch_size = batch_size
        self.mini_batch = mini_batch
        self.save = save
        self.load = load
        self.path = path

class Layer_Qparam:
    
    def __init__(self,
                input_qparam: QParam,
                act_qparam: QParam,
                wt_qparam: QParam,
                 ):
        self.input_qparam = input_qparam
        self.act_qparam = act_qparam
        self.wt_qparam = wt_qparam


def fillx(x, blk_size):
    n_item, vec_dim = x.size()
    reside = vec_dim % blk_size
    if (reside != 0):
        pad = ZeroPad2d(padding=(0,blk_size - reside,0,0))
        x = pad(x)
    print(x.size())
    return x

def plot_x(data, enable=False):
    if enable:
#        data_np = data.cpu()
        data_np = data.detach().cpu().numpy()
        sns.heatmap(data = data_np,
                    cmap = 'PuBuGn', 
#                    vmin=0.0,
#                    vmax=22,
        #            linewidths = 0.0001, 
        #            annot = False, 
        #            fmt = '.1e' 
                    )
        plt.title('heatmap')
        plt.show()
        
def sort_by_deg(x, edge_index):
    deg = torch_scatter.scatter_add(
            torch.ones((edge_index.size(1), ), 
            device=edge_index.device), 
            edge_index[1], 
            dim=0, 
            dim_size=x.size()[0]
            )
    index = deg.sort()[1]
    y = torch.index_select(x, dim=0, index=index)
    return y, deg, index

#def groupedBydegree(deg, group_bound):
#    deg_s, index = deg.sort()
#    fill = torch.zeros_like(deg)
#    deg_group = []
#    low_bound = 0
#    for bound in group_bound:
#        deg_gp = torch.where((deg_s > low_bound & deg_s <= bound), deg_s, fill)
#        deg_gp = deg_gp.nonzero()
#        deg_gp = deg_gp.flatten(0)
#        deg_group.append(deg_gp)
#        low_bound = bound+1
#    return deg_group, index
    
def groupedBydegree(deg, group_bound, x):
    deg_s, index = deg.sort()
    deg_group = []
    low_bound = 0
    index_group = []
    data_group = []
    for bound in group_bound:
        mask_hi = deg_s <= bound
        mask_lo = deg_s >= low_bound
        mask = mask_lo & mask_hi
        mask = mask.nonzero().flatten(0)
        deg_gp = deg_s.index_select(0, mask)
        deg_group.append(deg_gp)
        index_gp = index.index_select(0, mask)
        index_group.append(index_gp)
        low_bound = bound+1
    for idx in index_group:
        data_gp = torch.index_select(x, dim=0, index=idx)
        data_group.append(data_gp)
    return deg_group, index_group, data_group

import numpy as np
import scipy.sparse as sp
def CSR(x):
    print('Apply CSR...')
    h, w = x.size()
    A = np.array(x)
    AS = sp.csr_matrix(A)
    print("data=",AS.data, AS.data.shape)
    print("indptr=",AS.indptr, AS.indptr.shape)
    print("indices=",AS.indices, AS.indices.shape)
    print("compress_rate=",(len(AS.data)+len(AS.indptr)+len(AS.indices))/(w*h))
    return AS.data, AS.indptr, AS.indices

def sizeCsr(x):
#    data, ptr, idx = CSR(x)
    memsiz = 0
    for i in CSR(x):
        memsiz += (int(math.log(i.max(), 2))/8) * (len(i))
    return memsiz

import os
def fetchAssign(input_qparam:QParam, n_nodes, device):
    if input_qparam.mini_batch:
        assignments = torch.load(os.path.join(input_qparam.path, f"blksize_{input_qparam.block_size}_batchsiz_{input_qparam.batch_size}_ncents_{input_qparam.n_centroids}_assignments.pth"), map_location=device)
        centroids = torch.load(os.path.join(input_qparam.path, f"blksize_{input_qparam.block_size}_batchsiz_{input_qparam.batch_size}_ncents_{input_qparam.n_centroids}_centroids.pth"), map_location=device)
    else:
        assignments = torch.load(os.path.join(input_qparam.path, f"blksize_{input_qparam.block_size}_assignments.pth"), map_location=device)
        centroids = torch.load(os.path.join(input_qparam.path, f"blksize_{input_qparam.block_size}_centroids.pth"), map_location=device)
    assignments = assignments.reshape(n_nodes, -1)
    return assignments, centroids

def result_container(qnt_param, *args, **kwargs):
    x = sizeTracker(qnt_param, *args, **kwargs)
    y = PrettyTable(['accuracy', 'mean(%)', 'lower boundary(%)', 'upper boundary(%)'])
    y.align['accuracy'] = '1'
#    x.padding_width = 1
    y.add_row(["accuracy", f"{kwargs['mean']:.4f}", f"{kwargs['lo']:.4f}", f"{kwargs['hi']:.4f}"])
    print(y)
    return x,y


def sizeTracker(qnt_param, *args, **kwargs):
    layers_size = []
    for layer_size in args:
        layers_size.append(layer_size)   
        
    qparams = []
    input_size_before = []
    input_size_after = []
    input_compress_ratio = []
    for _, qparam in qnt_param.items():
        qparams.append(qparam) 
        
    for i in range(len(qnt_param)):
        input_size_b, input_size_a, compression_ratio = inputSize(qparams[i].input_qparam, qparams[i].act_qparam, layers_size[i], **kwargs) 
        input_size_before.append(input_size_b)
        input_size_after.append(input_size_a)
        input_compress_ratio.append(compression_ratio)
        
    model_size_b, model_size_a, model_compress_ratio = modelSize(qparams, **kwargs)
    total_size_b = (model_size_b + input_size_before[0] +  input_size_before[1])/(1024**2)
    total_size_a = (model_size_a + input_size_after[0] + input_size_after[1])/(1024**2)
    model_size_a /= (1024**2)
    model_size_b /= (1024**2)
    input_size_after[0] /= (1024**2)
    input_size_after[1] /= (1024**2)
    input_size_before[0] /= (1024**2)
    input_size_before[1] /= (1024**2)
    overall_compress_ratio = total_size_b/total_size_a
#    print(f'overall size before quantization: {total_size_b}B({total_size_b/1048576:.4f}MB),\
#          after quantization: {total_size_a}B({total_size_a/1048576:.4f}MB), \
#          compression ratio is {total_size_b/total_size_a:.4f}\n')
    x = PrettyTable(['data type', 'size before quant(MB)', 'size after quant(MB)', 'compression ratio'])
    x.align['data type'] = '1'
#    x.padding_width = 1
    x.add_row(["input_layer1", f"{input_size_before[0]:.4f}", f"{input_size_after[0]:.4f}", f"{input_compress_ratio[0]:.4f}"])
    x.add_row(["input_layer2", f"{input_size_before[1]:.4f}", f"{input_size_after[1]:.4f}", f"{input_compress_ratio[1]:.4f}"])
    x.add_row(["model_size", f"{model_size_b:.4f}", f"{model_size_a:.4f}", f"{model_compress_ratio:.4f}"])
    x.add_row(["overall_size", f"{total_size_b:.4f}", f"{total_size_a:.4f}", f"{overall_compress_ratio:.4f}"])
    print(x)
    return x
    

def inputSize(input_qnt_param, act_qnt_param, input_size, **kwargs):
    n_nodes, n_feature = input_size
    input_size_b = n_nodes*n_feature*4
    if input_qnt_param.pqnt:
        cents_size = centroids_size(n_nodes, n_feature, 
                                        input_qnt_param.block_size, 
                                        input_qnt_param.n_centroids, 
                                        input_qnt_param.batch_size, 
                                        input_qnt_param.mini_batch
                                    )
        if act_qnt_param.sqnt:
            cents_size/=(32/act_qnt_param.bits[0])
            
        ass_size = assignments_size(n_nodes, n_feature, 
                                        input_qnt_param.block_size, 
                                        input_qnt_param.n_centroids, 
                                        input_qnt_param.batch_size, 
                                        input_qnt_param.mini_batch
                                    )
        if kwargs['CSR'] == True:
            ass_size = sizeCsr(kwargs['assignments'])
        input_size_a = ass_size + cents_size
#        print(f'cents_siz:{cents_size}, ass_siz:{ass_size}')
    else:
        if act_qnt_param.sqnt:
            input_size_a = input_size_b/(32/act_qnt_param.bits[0])
        else:
            input_size_a = input_size_b
    
    compress_ratio = input_size_b / input_size_a

#    print(f"input size before quantization: {input_size_b}B({input_size_b/1048576:.4f}MB), \
#          after quantization: {input_size_a}B({input_size_a/1048576:.4f}MB), \
#          compression ratio is {input_size_b/input_size_a:.4f}\n")
    return input_size_b, input_size_a, compress_ratio
    

def modelSize(qnt_param, **kwargs):
    model = kwargs['model']
    if model == 'GCN':
        layer1_wt = kwargs['n_feature'] * kwargs['n_hid'] * 4
        layer2_wt = kwargs['n_class'] * kwargs['n_hid'] * 4
        model_size_b = layer1_wt + layer2_wt
        if qnt_param[0].wt_qparam.sqnt:
            layer1_wt /= (32/qnt_param[0].wt_qparam.bits[0])
        if qnt_param[1].wt_qparam.sqnt:
            layer2_wt /= (32/qnt_param[1].wt_qparam.bits[0])
        model_size_a = layer1_wt + layer2_wt
    elif model == 'GAT':
        in_channel_num = kwargs['in_channel_num']
        layer1_wt = in_channel_num * kwargs['n_feature'] * kwargs['n_heads'] * kwargs['n_hid'] * 4
        layer2_wt = in_channel_num * kwargs['n_class'] * kwargs['n_heads'] * kwargs['n_hid'] * 4
        model_size_b = layer1_wt + layer2_wt
        if qnt_param[0].wt_qparam.sqnt:
            layer1_wt /= (32/qnt_param[0].wt_qparam.bits[0])
        if qnt_param[1].wt_qparam.sqnt:
            layer2_wt /= (32/qnt_param[1].wt_qparam.bits[0])
        model_size_a = layer1_wt + layer2_wt
    compress_ratio = model_size_b / model_size_a
#    print(f"model size before quantization: {model_size_b}B({model_size_b/1048576:.4f}MB), after quantization: {model_size_a}B({model_size_a/1048576:.4f}MB), compression ratio is {model_size_b/model_size_a:.4f}\n")
    return model_size_b, model_size_a, compress_ratio   
        

def centroids_size(n_nodes, n_feature, blksiz, n_cents, batch_size=1024, mini_batch=False):
    if mini_batch:
        cents_siz = (n_nodes/batch_size)*n_cents*blksiz*4
        return cents_siz
    else:
        return (n_cents*4*blksiz)

def assignments_size(n_nodes, n_feature, blksiz, n_cents, batch_size=1024, mini_batch=False):
    if mini_batch:
        ass_siz = (int(math.log(n_nodes*n_cents/batch_size, 2))/8) * (n_nodes*n_feature/blksiz)
    else:
        ass_siz = (int(math.log(n_cents, 2))/8) * (n_nodes*n_feature/blksiz)
    return ass_siz

import subprocess
def avaliable_GPU():
    nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
    total_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
    total_GPU = total_GPU_str.split('\n')
    total_GPU = np.array([int(device_i) for device_i in total_GPU])
    avail_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
    avail_GPU = avail_GPU_str.split('\n')
    avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
    avail_GPU_ratio = avail_GPU / total_GPU
    sel_GPU = torch.tensor(avail_GPU_ratio).index_select(0, torch.arange(nDevice))
    return np.argmax(sel_GPU)