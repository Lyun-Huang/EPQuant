#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 08:58:58 2021

@author: root
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, nfeature, nclass):
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeature, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, nclass, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)