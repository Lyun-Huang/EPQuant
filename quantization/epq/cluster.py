#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:11:54 2021

@author: root
"""
import torch
import logging
import random
from collections import Counter

class kmeans():

    def __init__(
        self,
        x,
        n_clusters=32,
        iters=10,
        eps=1e-6,
        try_cluster=30,
    ):
        self.x = x
        self.n_clusters = n_clusters
        self.centers = torch.Tensor()
        self.labels = torch.Tensor()
        self.eps = eps
        self.try_cluster = try_cluster
        self.iters = iters

    def train(self):
        self.initClusters()
        for i in range(self.iters):
            self.clustering(i)

    def clustering(self, i):
        
        distances = self.compute_distances()  
        self.labels = torch.argmin(distances, dim=0)  
        n_empty_clusters = self.resolve_empty_clusters()

        for k in range(self.n_clusters):
            x_k = self.x[:, self.labels == k]  
            self.centers[k] = x_k.mean(dim=1)

        mse = (self.centers[self.labels].t() - self.x).norm(p=2).item()
        logging.info(f"Iteration: {i}, unresolved empty clusters: {n_empty_clusters}, mse: {mse:.4f}.")
            
            
    def initClusters(self):

        _, n_items = self.x.size()
        indices = torch.randint(low=0, high=n_items, size=(self.n_clusters,)).long()
        self.centers = self.x[:, indices].t()

    def resolve_empty_clusters(self):

        logging.debug(f"Resolve empty clusters...")
        # empty clusters
        counts = Counter(map(lambda x: x.item(), self.labels))
        empty_clusters = set(range(self.n_clusters)) - set(counts.keys())
        n_empty_clusters = len(empty_clusters)

        cnt = 0
        conti_no_change = False
        conti_cnt = 0
        last_n_empty_clusters = 0
        while len(empty_clusters) > 0:
            
            last_n_empty_clusters = len(empty_clusters)
            k = random.choice(list(empty_clusters))
            m = counts.most_common(1)[0][0]
            e = torch.randn_like(self.centers[m]) * self.eps
            self.centers[k] = self.centers[m].clone()
            self.centers[k] += e
            self.centers[m] -= e

            # recompute labels
            distances = self.compute_distances()
            self.labels = torch.argmin(distances, dim=0) 

            counts = Counter(map(lambda x: x.item(), self.labels))
            
            if len(empty_clusters) == last_n_empty_clusters:
                conti_cnt += 1
#            print(conti_cnt)
            
            if conti_cnt == 10:
                conti_no_change = True
            
            if cnt == self.try_cluster or conti_no_change:
                logging.warning(
                    f"Could not resolve all empty clusters, {len(empty_clusters)} remaining"
                )
                self.codebook_resort()
                break
            cnt += 1
            
            logging.debug(f"{len(empty_clusters)} empty clusters remaining")

        logging.debug(f"Resolve empty clusters Done")
        return n_empty_clusters
    
    def codebook_resort(self):
        used_centroids = self.labels.unique()  #get the centers index already assigned
        reduced_centroids = self.centers[used_centroids]
        self.centers = reduced_centroids #get new codebook
        self.n_clusters = self.centers.size()[0]  #update number of centers
        mapping = {}
        for i, j in enumerate(used_centroids):
            mapping[str(int(j.item()))] = i
        for i, j in enumerate(self.labels):
            self.labels[i] = mapping[str(int(j.item()))]

    def compute_distances(self):

        distance = torch.cat(
            [
                (self.x[None, :, :] - centroids_c[:, :, None]).float().norm(p=2, dim=1)
                for centroids_c in self.centers.chunk(1, dim=0)
            ],
            dim=0,
        )
        return distance


