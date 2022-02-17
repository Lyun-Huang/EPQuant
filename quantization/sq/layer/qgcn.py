#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:15:46 2021

@author: root
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, PairTensor
from torch import Tensor
import numpy as np
from torch_geometric.nn import GCNConv
from quantization.sq.ops import intqnt
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from quantization.utils import Layer_Qparam

class IntGCNConv(GCNConv):
    
    def __init__(self, qparam: Layer_Qparam, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        super(IntGCNConv, self).__init__(
                in_channels,
                out_channels,
                improved,
                cached,
                add_self_loops,
                normalize,
                bias,
                **kwargs)
        # quantization parameters
        self.wt_qparam = qparam.wt_qparam
        self.act_qparam = qparam.act_qparam
        self.counter = 0

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
                       
        # train with QuantNoise and evaluate the fully quantized network
        if self.wt_qparam.sqnt:
            p = self.wt_qparam.p if self.training else 1
    
            # update parameters every 100 iterations
            if self.counter % self.wt_qparam.update_step == 0:
                self.wt_scale = None
                self.wt_zero_point = None
    
            # quantize weight
            weight_quantized, self.wt_scale, self.wt_zero_point = intqnt(
                self.lin.weight.detach(),
                bits=self.wt_qparam.bits[0],
                method=self.wt_qparam.method,
                scale=self.wt_scale,
                zero_point=self.wt_zero_point,
            )
    
            # mask to apply noise
            mask = torch.zeros_like(self.lin.weight)
            mask.bernoulli_(1 - p)
            noise = (weight_quantized - self.lin.weight).masked_fill(mask.bool(), 0)
    
            # using straight-through estimator (STE)
            clamp_low = -self.wt_scale * self.wt_zero_point
            clamp_high = self.wt_scale * (2 ** self.wt_qparam.bits[0] - 1 - self.wt_zero_point)
            self.lin.weight.data = (
                torch.clamp(self.lin.weight, clamp_low.item(), clamp_high.item())
                + noise.detach()
            )
        
        if self.act_qparam.sqnt:
            # update parameters every 100 iterations
            if self.counter % self.act_qparam.update_step == 0:
                self.scale_com = None
                self.zero_point_com = None
                self.scale_agg = None
                self.zero_point_agg = None
            
            #quantize combination
            x_quantized, self.scale_com, self.zero_point_com = intqnt(
                x.detach(),
                bits=self.act_qparam.bits[0],
                method=self.act_qparam.method,
                scale=self.scale_com,
                zero_point=self.zero_point_com,
            )
            clamp_low = -self.scale_com * self.zero_point_com
            clamp_high = self.scale_com * (2 ** self.act_qparam.bits[0] - 1 - self.zero_point_com)
            x = torch.clamp(x, clamp_low.item(), clamp_high.item())

        x = self.lin(x)

        if self.act_qparam.sqnt:
            #quantize aggregation
            x_quantized, self.scale_agg, self.zero_point_agg = intqnt(
                x.detach(),
                bits=self.act_qparam.bits[1],
                method=self.act_qparam.method,
                scale=self.scale_agg,
                zero_point=self.zero_point_agg,
            )
            clamp_low = -self.scale_agg * self.zero_point_agg
            clamp_high = self.scale_agg * (2 ** self.act_qparam.bits[1] - 1 - self.zero_point_agg)
            x = torch.clamp(x, clamp_low.item(), clamp_high.item())
        
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias
        
        self.counter += 1
        
        return out


#    def extra_repr(self):
#        return (
#            "in_channels={}, out_channels={}, bias={}, quant_noise={}, "
#            "bits={}, method={}".format(
#                self.in_channels,
#                self.out_channels,
#                self.bias is not None,
#                self.p,
#                self.bits,
#                self.method,
#            )
#        )
    
#model=IntGCNConv(in_channels=2, out_channels=3, bias=True, bits=(8,8,8))
#print(model.bias.data)
