#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch

#from .utils import plot_x
from quantization.epq.pq import EPQ
import numpy as np
import time
import os
from quantization.utils import QParam, plot_x, fillx

class ActPQ():

    def __init__(
        self,
        module,
        module_name,
        n_centroids,
        block_size,
        n_iter=15,
        eps=1e-6,
        try_cluster=100,
        path='',
        suffix='',
        layer='',
        batch_size=1,
        mini_batch=False,
        register=False,
        save=False,
        load=False,
        reduced=False,
    ):
        self.module = module
        self.handle = []
        self.register_hook(register)
        self.save = save
        self.load = load
        self.module_name = module_name
        self.n_centroids = n_centroids
        self.block_size = block_size
        self.n_iter = n_iter
        self.eps = eps
        self.try_cluster = try_cluster
        self.path = path
        self.suffix = suffix
        self.layer = layer
        self.mini_batch = mini_batch
        self.batch_size = batch_size
        self.reduced = reduced
        
    def input_quant_multi(self, x, device_list):
        x_q = x
#        plot_x(x_q[:20, :100], True)
        self.centroids = torch.Tensor()
        self.assignments = torch.Tensor()
        if self.save:
            print(f"\nStart Product Quantization, input data size is {x_q.size()}, current time is \
                {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
            
        if self.mini_batch and self.save:
            for i, xq in enumerate(x_q.chunk(int(x_q.size()[0]/self.batch_size), dim=0)):
                quantizer = EPQ(
                    xq,
                    block_size=self.block_size,
                    n_centroids=self.n_centroids,
                    n_iter=self.n_iter,
                    eps=self.eps,
                    try_cluster=self.try_cluster,
                    simplify=self.reduced,
                )
                print(f"Product Quant for batch number {i}/{int(x_q.size()[0]/self.batch_size)}, \
                          {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
                quantizer.quantize()
                
#                if i == 0:
#                    torch.save(quantizer.centroids, os.path.join(self.path, f"batch_{i}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_centroids.pth"))
#                    torch.save(quantizer.assignments, os.path.join(self.path, f"batch_{i}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_assignments.pth"))
                
                if self.assignments.nelement() == 0:
                    self.assignments = quantizer.assignments.contiguous()
                else:
                    reorder_ass = quantizer.assignments.contiguous() + self.centroids.size()[0]
                    self.assignments = torch.cat([self.assignments, reorder_ass], dim=0)
                
                if self.centroids.nelement() == 0:
                    self.centroids = quantizer.centroids.contiguous()
                else:
                    self.centroids = torch.cat([self.centroids, quantizer.centroids.contiguous()], dim=0)
                    
                max_batch_num = i
                    
#                x_dq = quantizer.dequant()
#                torch.save(x_dq, os.path.join(self.path, f"batch_{i}_blksize_{self.block_size}_xdq.pth"))
                
        else:
            self.quantizer = EPQ(
                x_q,
                block_size=self.block_size,
                n_centroids=self.n_centroids,
                n_iter=self.n_iter,
                eps=self.eps,
                try_cluster=self.try_cluster,
                simplify=self.reduced,
            )

            if self.save:
                self.quantizer.quantize()
                self.centroids = self.quantizer.centroids.contiguous()
                self.assignments = self.quantizer.assignments.contiguous()
                
                
        if self.save: 
            if self.mini_batch:            
                torch.save(self.centroids, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_centroids.pth"))
                torch.save(self.assignments, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_assignments.pth"))
            else:
                torch.save(self.centroids, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_centroids.pth"))
                torch.save(self.assignments, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_assignments.pth"))
            print(f"\nEnd Product Quantization, codebook size is {self.centroids.size()}, \
                assignment size is {self.assignments.size()}, \
                current time is {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
        
        if self.load:
            if self.mini_batch:
                self.quantizer.load(os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}"), device=x.device)
            else:
                self.quantizer.load(os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}"), device=x.device)
            self.centroids = self.quantizer.centroids.contiguous()
            self.assignments = self.quantizer.assignments.contiguous()
            
        
        if self.mini_batch and self.save:
            self.quantizer = EPQ(
                x_q,
                block_size=self.block_size,
                n_centroids=self.n_centroids,
                n_iter=self.n_iter,
                eps=self.eps,
                try_cluster=self.try_cluster,
                simplify=False if self.save else False,
            )
            self.quantizer.load(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}", device=x.device)
        
        #return quantized tensor
        x_q = self.quantizer.dequant()
#        plot_x(x_q[:20, :100], True)
        x = x_q
        return x
        
    def input_quant(self, x):
        x_q = x
#        plot_x(x_q[:20, :100], True)
        self.centroids = torch.Tensor()
        self.assignments = torch.Tensor()
        if self.save:
            print(f"\nStart Product Quantization, input data size is {x_q.size()}, current time is \
                {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
            
        if self.mini_batch and self.save:
            for i, xq in enumerate(x_q.chunk(int(x_q.size()[0]/self.batch_size), dim=0)):
                quantizer = EPQ(
                    xq,
                    block_size=self.block_size,
                    n_centroids=self.n_centroids,
                    n_iter=self.n_iter,
                    eps=self.eps,
                    try_cluster=self.try_cluster,
                    simplify=False if self.save else False,
                )
                print(f"Product Quant for batch number {i}/{int(x_q.size()[0]/self.batch_size)}, \
                          {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
                quantizer.quantize()
                
#                if i == 0:
#                    torch.save(quantizer.centroids, os.path.join(self.path, f"batch_{i}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_centroids.pth"))
#                    torch.save(quantizer.assignments, os.path.join(self.path, f"batch_{i}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_assignments.pth"))
                
                if self.assignments.nelement() == 0:
                    self.assignments = quantizer.assignments.contiguous()
                else:
                    reorder_ass = quantizer.assignments.contiguous() + self.centroids.size()[0]
                    self.assignments = torch.cat([self.assignments, reorder_ass], dim=0)
                
                if self.centroids.nelement() == 0:
                    self.centroids = quantizer.centroids.contiguous()
                else:
                    self.centroids = torch.cat([self.centroids, quantizer.centroids.contiguous()], dim=0)
                    
                max_batch_num = i
                    
#                x_dq = quantizer.dequant()
#                torch.save(x_dq, os.path.join(self.path, f"batch_{i}_blksize_{self.block_size}_xdq.pth"))
                
        else:
            self.quantizer = EPQ(
                x_q,
                block_size=self.block_size,
                n_centroids=self.n_centroids,
                n_iter=self.n_iter,
                eps=self.eps,
                try_cluster=self.try_cluster,
                simplify=True if self.save else False,
            )

            if self.save:
                self.quantizer.quantize()
                self.centroids = self.quantizer.centroids.contiguous()
                self.assignments = self.quantizer.assignments.contiguous()
                
                
        if self.save: 
            if self.mini_batch:            
                torch.save(self.centroids, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_centroids.pth"))
                torch.save(self.assignments, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_assignments.pth"))
            else:
                torch.save(self.centroids, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_centroids.pth"))
                torch.save(self.assignments, os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_assignments.pth"))
            print(f"\nEnd Product Quantization, codebook size is {self.centroids.size()}, \
                assignment size is {self.assignments.size()}, \
                current time is {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}")
        
        if self.load:
            if self.mini_batch:
                self.quantizer.load(os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}"), device=x.device)
            else:
                self.quantizer.load(os.path.join(self.path, f"{self.suffix}blksize_{self.block_size}"), device=x.device)
            self.centroids = self.quantizer.centroids.contiguous()
            self.assignments = self.quantizer.assignments.contiguous()
            
        
        if self.mini_batch and self.save:
            self.quantizer = EPQ(
                x_q,
                block_size=self.block_size,
                n_centroids=self.n_centroids,
                n_iter=self.n_iter,
                eps=self.eps,
                try_cluster=self.try_cluster,
                simplify=False if self.save else False,
            )
            self.quantizer.load(self.path, f"{self.suffix}blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}", device=x.device)
        
        #return quantized tensor
        x_q = self.quantizer.dequant()
#        plot_x(x_q[:20, :100], True)
        x = x_q
        return x
    
    

    def register_hook(self, register=False):
        
        def input_quantize_hook(module, x):
            x_l = list(x)           #transform tuple to list
            x_q = x_l[0] 
#            print(x_q.size())
#            print(f"Layer {module}.input beforce PQ:")
#            plot_x(x_q[:20, :100], True)
            return self.input_quant(x_q)

        if register:
            self.handle.append(self.module.register_forward_pre_hook(input_quantize_hook))
