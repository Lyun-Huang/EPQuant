#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 06:18:48 2021

@author: root
"""

from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import SAGEConv
from quantization.utils import Layer_Qparam
from quantization.sq.ops import intqnt

class IntSAGEConv(SAGEConv):

    def __init__(self, qparam: Layer_Qparam, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        super(IntSAGEConv, self).__init__(
                    in_channels,
                    out_channels,
                    normalize,
                    bias,
                    **kwargs,
                )
        # quantization parameters
        self.wt_qparam = qparam.wt_qparam
        self.act_qparam = qparam.act_qparam
        self.counter = 0


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

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
                    self.scale_l_com = None
                    self.zero_point_l_com = None
                    self.scale_r_com = None
                    self.zero_point_r_com = None
                    self.scale_agg = None
                    self.zero_point_agg = None
                
                #quantize combination
                x_quantized, self.scale_agg, self.zero_point_agg = intqnt(
                    x.detach(),
                    bits=self.act_qparam.bits[0],
                    method=self.act_qparam.method,
                    scale=self.scale_agg,
                    zero_point=self.zero_point_agg,
                )
                clamp_low = -self.scale_agg * self.zero_point_agg
                clamp_high = self.scale_agg * (2 ** self.act_qparam.bits[0] - 1 - self.zero_point_agg)
                x = torch.clamp(x, clamp_low.item(), clamp_high.item())
            else:
#                raise NotImplementedError
                # update parameters every 100 iterations
                if self.counter % self.act_qparam.update_step == 0:
                    self.scale_l_com = None
                    self.zero_point_l_com = None
                    self.scale_r_com = None
                    self.zero_point_r_com = None
                    self.scale_agg = None
                    self.zero_point_agg = None
                
                #quantize combination
                _, self.scale_agg, self.zero_point_agg = intqnt(
                    x[0].detach(),
                    bits=self.act_qparam.bits[0],
                    method=self.act_qparam.method,
                    scale=self.scale_agg,
                    zero_point=self.zero_point_agg,
                )
                clamp_low = -self.scale_agg * self.zero_point_agg
                clamp_high = self.scale_agg * (2 ** self.act_qparam.bits[0] - 1 - self.zero_point_agg)
                x_0 = torch.clamp(x[0], clamp_low.item(), clamp_high.item())
                x_1 = torch.clamp(x[1], clamp_low.item(), clamp_high.item())
                x = tuple([x_0, x_1])

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        
        if self.act_qparam.sqnt:       
            #quantize combination
            _, self.scale_l_com, self.zero_point_l_com = intqnt(
                out.detach(),
                bits=self.act_qparam.bits[1],
                method=self.act_qparam.method,
                scale=self.scale_l_com,
                zero_point=self.zero_point_l_com,
            )
            clamp_low = -self.scale_l_com * self.zero_point_l_com
            clamp_high = self.scale_l_com * (2 ** self.act_qparam.bits[1] - 1 - self.zero_point_l_com)
            out = torch.clamp(out, clamp_low.item(), clamp_high.item())

        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            if self.act_qparam.sqnt:              
                #quantize combination
                _, self.scale_r_com, self.zero_point_r_com = intqnt(
                    x_r.detach(),
                    bits=self.act_qparam.bits[1],
                    method=self.act_qparam.method,
                    scale=self.scale_r_com,
                    zero_point=self.zero_point_r_com,
                )
                clamp_low = -self.scale_r_com * self.zero_point_r_com
                clamp_high = self.scale_r_com * (2 ** self.act_qparam.bits[1] - 1 - self.zero_point_r_com)
                x_r = torch.clamp(x_r, clamp_low.item(), clamp_high.item())
            
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

