#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import os
#from sklearn.cluster import KMeans
from quantization.epq.cluster import kmeans
import torch.nn as nn
from quantization.utils import fillx

class EPQ(nn.Module):

    def __init__(
        self,
        x,
        block_size=4,
        n_centroids=256,
        n_iter=10,
        try_cluster=15,
        eps=1e-6,       
        simplify=False,
#        train=False,
#        restore=False,
    ):
        self.x = x
        self.block_size = block_size
        self.simplify = simplify
        self.index = None
        self.n_centroids = n_centroids
        self.n_iter = n_iter
        self.try_cluster = try_cluster
        self.eps = eps
        self.n_nodes, self.dimension = x.size()

    def split(self):
        
        #Given block size, split the input embedding table to a set of subvectors of size block_size.
#        self.n_nodes, self.dimension = self.x.size()
        if self.dimension % self.block_size != 0:
            self.x = fillx(self.x, self.block_size)
            
        self.x = (
            self.x.reshape(self.n_nodes, -1, self.block_size)  #(n_nodes, dimension/block_size, block_size)
            .permute(2, 0, 1)   #(block_size, n_nodes, dimension/block_size)
            .flatten(1, 2)      #(block_size, n_nodes*dimension/block_size )
        )

    def xstrip(self):
        print("simplified PQ...")
        abs_add = torch.sum(self.x.absolute(), dim=0)
        abs_add_s, index_s = abs_add.sort(descending=True)
        split_point = None
        for i in range(len(abs_add_s)):
            if abs_add_s[i] == 0:
                split_point = i
                break
        w_s = torch.index_select(self.x, dim=1, index=index_s)
        if split_point != None:
            self.x = w_s[:, :split_point]
        else:
            self.x = w_s
#        print(w.size())
#        print(w)
#        print(index_s)
        self.index = index_s
        
        
    def recover(self):
        zero_centroid = torch.zeros([self.block_size]).to(self.centroids.device)
        self.centroids = torch.cat([zero_centroid[None, :], self.centroids], dim=0)
        left_assignments=torch.zeros([len(self.index) - self.x.size()[1]]).to(self.assignments.device)
        self.assignments = torch.cat([self.assignments+1, left_assignments]).type_as(self.assignments)
        self.new_assignments = self.assignments.clone()
        self.assignments[self.index] = self.new_assignments 
        self.assignments = self.assignments.long()
            
    def quantize(self):
        
        print(self.x.size())
        self.split()
        print(self.x.size())
        if self.simplify:
            self.xstrip()
            print(self.x.size())

        km = kmeans(
            self.x,
            n_clusters=self.n_centroids,
            iters=self.n_iter,
            eps=self.eps,
            try_cluster=self.try_cluster,
        )
        
        km.train()
        self.centroids = km.centers
        self.n_centroids = km.n_clusters
        self.assignments = km.labels
        
        if self.simplify:
            self.recover()
        
    def dequant(self):
        
#        self.n_nodes, self.dimension = self.x.size()
        x = self.centroids[self.assignments].reshape(self.n_nodes, -1, self.block_size).flatten(1, 2)
        return x[:, :self.dimension]
        
    
    def save(self, suffix):
        torch.save(self.centroids, f"{suffix}_centroids.pth")
        torch.save(self.assignments, f"{suffix}_assignments.pth")

    def load(self, suffix, device):
        self.centroids = torch.load(f"{suffix}_centroids.pth", map_location=device)
        self.assignments = torch.load(f"{suffix}_assignments.pth", map_location=device)
