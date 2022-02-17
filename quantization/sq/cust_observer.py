#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

import torch
from torch.quantization.observer import HistogramObserver

class IntHistogramObserver(HistogramObserver):

    def __init__(self, bits=8, signed=False):
        super(IntHistogramObserver, self).__init__()
        self.bits = bits
        self.signed = signed

    def _calculate_qparams(self, min_val, max_val):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        r"""Calculates the quantization parameters, given min and max
        value tensors. Works for both per tensor and per channel cases
        Args:
            min_val: Minimum values per channel
            max_val: Maximum values per channel
        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """
        if min_val.numel() == 0 or max_val.numel() == 0:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        if min_val.dim() == 0 or max_val.dim() == 0:
            assert min_val <= max_val, "min {} should be less than max {}".format(
                min_val, max_val
            )
        else:
            assert torch.sum(min_val <= max_val) == len(min_val), "min {} should be less than max {}".format(
                min_val, max_val
            )

#        if self.dtype == torch.qint8:
#            if self.reduce_range:
#                qmin, qmax = -64, 63
#            else:
#                qmin, qmax = -128, 127
#        elif self.dtype == torch.quint8:
#            if self.reduce_range:
#                qmin, qmax = 0, 127
#            else:
#                qmin, qmax = 0, 255
#        else:
        if self.signed:
            qmin = - 2. ** (self.bits - 1)
            qmax = 2. ** (self.bits - 1) - 1
        else:
            qmin, qmax = 0, 2.**self.bits - 1.

        min_val = torch.min(min_val, torch.zeros_like(min_val))
        max_val = torch.max(max_val, torch.zeros_like(max_val))

        scale = torch.ones(min_val.size(), dtype=torch.float32)
        zero_point = torch.zeros(min_val.size(), dtype=torch.int64)
        device = 'cuda' if min_val.is_cuda else 'cpu'

        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale = torch.max(scale, torch.tensor(self.eps, device=device, dtype=scale.dtype))
            if self.dtype == torch.quint8:
                zero_point = zero_point.new_full(zero_point.size(), 128)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
#            scale = torch.max(scale, torch.tensor(self.eps, device=device, dtype=scale.dtype))
            scale = torch.max(scale, self.eps.clone().to(device))
            zero_point = qmin - torch.round(min_val / scale)
            zero_point = torch.max(zero_point, torch.tensor(qmin, device=device, dtype=zero_point.dtype))
            zero_point = torch.min(zero_point, torch.tensor(qmax, device=device, dtype=zero_point.dtype))


        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype)

        return scale, zero_point