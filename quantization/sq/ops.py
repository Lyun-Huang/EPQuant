#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
from quantization.sq.cust_observer import IntHistogramObserver

def intqnt(w, bits, method, scale=None, zero_point=None):
    #q = globals()[f"int_{method}"]
    #return q(w, bits, scale=scale, zero_point=zero_point)
    return int_histogram(w, bits, scale=scale, zero_point=zero_point)


def quantize(w, bits, scale, zero_point, signed=False):
    if signed:
        qmin = - 2. ** (bits - 1)
        qmax = 2. ** (bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2.**bits - 1.
    return (
        torch.clamp(torch.round(w / scale + zero_point), qmin, qmax) - zero_point
    ) * scale


def int_histogram(w, bits, scale=None, zero_point=None):
    if scale is None:
        obs = IntHistogramObserver(bits=bits)
        obs = obs.cuda()
        _ = obs(w.float())
        scale, zero_point = obs.calculate_qparams()
        scale = scale.cuda().type_as(w)
        zero_point = zero_point.cuda().type_as(w)
    return quantize(w, bits, scale, zero_point), scale, zero_point
