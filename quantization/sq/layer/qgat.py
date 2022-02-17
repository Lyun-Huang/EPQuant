#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 03:07:03 2021

@author: root
"""

from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from quantization.utils import Layer_Qparam
from quantization.sq.ops import intqnt

class IntGATConv(GATConv):

    def __init__(self, qparam: Layer_Qparam, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        
        super(IntGATConv, self).__init__(
                    in_channels,
                    out_channels,
                    heads,
                    concat,
                    negative_slope,
                    dropout,
                    add_self_loops,
                    bias,
                    **kwargs
                )
        # quantization parameters
        self.wt_qparam = qparam.wt_qparam
        self.act_qparam = qparam.act_qparam
        self.counter = 0

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        
        if self.wt_qparam.sqnt:
            p = self.wt_qparam.p if self.training else 1
    
            # update parameters every 100 iterations
            if self.counter % self.wt_qparam.update_step == 0:
                self.wt_l_scale = None
                self.wt_l_zero_point = None
                self.wt_r_scale = None
                self.wt_r_zero_point = None
    
            # quantize weight
            weight_l_quantized, self.wt_l_scale, self.wt_l_zero_point = intqnt(
                self.lin_l.weight.detach(),
                bits=self.wt_qparam.bits[0],
                method=self.wt_qparam.method,
                scale=self.wt_l_scale,
                zero_point=self.wt_l_zero_point,
            )
            
            weight_r_quantized, self.wt_r_scale, self.wt_r_zero_point = intqnt(
                self.lin_r.weight.detach(),
                bits=self.wt_qparam.bits[0],
                method=self.wt_qparam.method,
                scale=self.wt_r_scale,
                zero_point=self.wt_r_zero_point,
            )
    
            # mask to apply noise
            mask_l = torch.zeros_like(self.lin_l.weight)
            mask_l.bernoulli_(1 - p)
            noise_l = (weight_l_quantized - self.lin_l.weight).masked_fill(mask_l.bool(), 0)
            mask_r = torch.zeros_like(self.lin_r.weight)
            mask_r.bernoulli_(1 - p)
            noise_r = (weight_r_quantized - self.lin_r.weight).masked_fill(mask_r.bool(), 0)
    
            # using straight-through estimator (STE)
            clamp_low = -self.wt_l_scale * self.wt_l_zero_point
            clamp_high = self.wt_l_scale * (2 ** self.wt_qparam.bits[0] - 1 - self.wt_l_zero_point)
            self.lin_l.weight.data = (
                torch.clamp(self.lin_l.weight, clamp_low.item(), clamp_high.item())
                + noise_l.detach()
            )
            clamp_low = -self.wt_r_scale * self.wt_r_zero_point
            clamp_high = self.wt_r_scale * (2 ** self.wt_qparam.bits[0] - 1 - self.wt_r_zero_point)
            self.lin_r.weight.data = (
                torch.clamp(self.lin_r.weight, clamp_low.item(), clamp_high.item())
                + noise_r.detach()
            )
        
        if self.act_qparam.sqnt:
            if isinstance(x, Tensor):
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
            else:
                raise NotImplementedError
        
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                num_nodes = size[1] if size is not None else num_nodes
                num_nodes = x_r.size(0) if x_r is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)


        if self.act_qparam.sqnt:
            if isinstance(x, Tensor):
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
            else:
                raise NotImplementedError
                
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

