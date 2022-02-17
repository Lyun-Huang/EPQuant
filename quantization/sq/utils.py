#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import logging
import re
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from operator import attrgetter, itemgetter
from quantization.utils import Layer_Qparam
from quantization.sq.layer.qact import ActQuant
from quantization.sq.layer.qgcn import IntGCNConv
from quantization.sq.layer.qgat import IntGATConv
from quantization.sq.layer.qsage import IntSAGEConv

MAPPING = {
           GCNConv: IntGCNConv,
           GATConv: IntGATConv,
           SAGEConv: IntSAGEConv,
           }

def quant_framework(model, **kwargs):

    quantized_layers, _, model = get_layers(model)
    layer_idx=0
    qparams = []
    for data_type, qparam in kwargs.items():
        qparams.append(qparam)

    for layer in quantized_layers:

        # recover module
        module = attrgetter(layer)(model)
        
        # quantization params
        q_params = {
            "wt_qparam": qparams[layer_idx].wt_qparam, 
            "act_qparam": qparams[layer_idx].act_qparam, 
            "counter": 0,
        }
            
        if (qparams[layer_idx].wt_qparam.sqnt or qparams[layer_idx].act_qparam.sqnt):
            if isinstance(module, tuple(MAPPING.keys())):
                QuantizedModule = MAPPING[module.__class__]
                quantized_module = QuantizedModule.__new__(QuantizedModule)
                params = module.__dict__
                params.update(q_params)
                quantized_module.__dict__.update(params)
                logging.info(f"Module {module} has been replaced with {quantized_module}")

            else:
                logging.warning(f"Module {module} not yet supported for quantization")
                quantized_module = module
        else:
            quantized_module = module

        a_q = ActQuant(module=quantized_module, module_name=str(layer), qparam=qparams[layer_idx].input_qparam)

        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_module)
        
        layer_idx += 1

#    logging.info(f"After quantization: {quantized_layers}")
    return quantized_layers

def get_layers(model):
    all_layers = []
    all_modules = []
    for name, module in model._modules.items():
        if name == 'module':            #dataparallel
            model = module
            for name, module in model._modules.items():
                all_layers.append(name)
                all_modules.append(module)
        else:
            all_layers.append(name)
            all_modules.append(module)
        
    return all_layers, all_modules, model

def attrsetter(*items):
    def resolve_attr(obj, attr):
        attrs = attr.split(".")
        head = attrs[:-1]
        tail = attrs[-1]

        for name in head:
            obj = getattr(obj, name)
        return obj, tail

    def g(obj, val):
        for attr in items:
            resolved_obj, resolved_attr = resolve_attr(obj, attr)
            setattr(resolved_obj, resolved_attr, val)

    return g
