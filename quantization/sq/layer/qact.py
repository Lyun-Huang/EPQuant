#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch

from quantization.sq.ops import intqnt
from quantization.epq.pqact import ActPQ
from quantization.utils import QParam

class ActQuant:

    def __init__(
        self,
        qparam: QParam,
        module,
        module_name='',
        register=False,
    ):
        self.module = module
        self.module_name = module_name
        self.counter = 0
        self.handle = []
        self.register_hook()
        self.p = qparam.p
        self.update_step = qparam.update_step      
        self.bits = qparam.bits
        self.method = qparam.method
        self.clamp_threshold = qparam.clamp_threshold
        self.save = qparam.save
        self.load = qparam.load
        self.n_centroids = qparam.n_centroids
        self.block_size = qparam.block_size
        self.n_iter = qparam.n_iter
        self.eps = qparam.eps
        self.try_cluster = qparam.try_cluster
        self.path = qparam.path
        self.mini_batch = qparam.mini_batch
        self.batch_size = qparam.batch_size
        self.input_pq = qparam.pqnt
        self.input_sq = qparam.sqnt

    def register_hook(self):
        
        def input_quantize_hook(module, x):
            # update parameters
            if self.counter % self.update_step == 0:
                self.scale = None
                self.zero_point = None
            self.counter += 1

            # train with QuantNoise and evaluate the fully quantized network
            p = self.p if self.module.training else 1
            
            #transform tuple to list
            x_l = list(x)
            x = x_l[0]

            if self.input_pq:
                pact = ActPQ(
                            None, 
                            module_name=self.module_name, 
                            n_centroids=self.n_centroids, 
                            block_size=self.block_size, 
                            try_cluster=self.try_cluster, 
                            n_iter=self.n_iter,
                            eps=self.eps,
                            save=self.save,
                            path=self.path,
                            load=self.load,
                            mini_batch=self.mini_batch,
                            batch_size=self.batch_size,
                        )
                x = pact.input_quant(x)
             
            if self.input_sq:
            # quantize activations
                x_q, self.scale, self.zero_point = intqnt(
                    x.detach(),
                    bits=self.bits[0],
                    method=self.method,
                    scale=self.scale,
                    zero_point=self.zero_point,
                )
    
                # mask to apply noise
                mask = torch.zeros_like(x)
                mask.bernoulli_(1 - p)
                noise = (x_q - x).masked_fill(mask.bool(), 0)
    
                # using straight-through estimator (STE)
                clamp_low = -self.scale * self.zero_point
                clamp_high = self.scale * (2 ** self.bits[0] - 1 - self.zero_point)
                x = torch.clamp(x, clamp_low.item(), clamp_high.item()) + noise.detach()
        
            
            #transform list back to tuple
            x_l[0] = x
            x = tuple(x_l)
            return x

        # register hook
        self.handle.append(self.module.register_forward_pre_hook(input_quantize_hook))