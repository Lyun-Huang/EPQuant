#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 07:37:02 2021

@author: root
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, nfeature, nhid, nclass, use_gdc=False):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(nfeature, nhid, cached=True,
                             normalize=not use_gdc)
        self.conv2 = GCNConv(nhid, nclass, cached=True,
                             normalize=not use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

