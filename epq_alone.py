import torch
import os.path as osp
import time, datetime
import os
import logging
import argparse
import torch_geometric.transforms as T
from multiprocessing import Process, Pool
from quantization.epq.pqact import ActPQ
from quantization.epq.pq import EPQ
from multiprocessing import freeze_support, Pool
import multiprocessing
import subprocess
import numpy as np
import gc

def argParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduce', action='store_true',
                        help='reduced PQ for sparse data.')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--pr_thread', type=int, default=5, help='number of parallel runs')
    parser.add_argument('--block_size', type=int, default=22)
    parser.add_argument('--ncents', type=int, default=64, 
                        help='the upper limit of learned clusters and the upper limit of learned clusters for each batch if mini_batch is true')
    parser.add_argument('--try_cluster', type=int, default=15,
                        help='number of attempts to find more centroids')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='number of iteration for cluster')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of nodes in each batch')
    parser.add_argument('--gpu_list', type=list, default=[0,2,6,7],
                        help='gpu device list')
    parser.add_argument('--path', type=str, default=f'./pq_multi/test/')
    # parser.add_argument('--f', type=str, default='result.txt', help='path of result file')
    
    args = parser.parse_args()
    print(args)
    return args

def dataProcess(dataset, fast=False, use_gdc=False):
    if dataset == 'Reddit':
        from torch_geometric.datasets import Reddit
        path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', 'Reddit')
        dataset = Reddit(path)
    elif dataset == 'Amazon2M':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name="ogbn-products", root='./data/')
    else:
        from torch_geometric.datasets import Planetoid
        path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', dataset)
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

    data = dataset[0]
    return data, dataset

class PR(ActPQ):

    def __init__(
        self,
        module,
        module_name,
        n_centroids,
        block_size,
        n_iter=15,
        eps=1e-6,
        try_cluster=100,
        path='',
        suffix='',
        layer='',
        batch_size=1,
        mini_batch=False,
        register=False,
        save=False,
        load=False,
        device_list=[],
        root_device=0,
        parallel_threads=10,
        reduced_PQ=True,
    ):
        super(PR, self).__init__(module, module_name, n_centroids, block_size, n_iter, eps, try_cluster,
                                 path, suffix, layer, batch_size, mini_batch, register, save, load)
        self.device_list = device_list
        self.root_device = root_device
        self.parallel_threads = parallel_threads
        self.reduced_PQ = reduced_PQ
        self.time_record = []


    def avaliable_GPU(self, space_reserve=0.3):
        # nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
        total_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total | grep -o '[0-9]\+'")
        total_GPU = total_GPU_str.split('\n')
        total_GPU = np.array([int(device_i) for device_i in total_GPU])
        avail_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
        avail_GPU = avail_GPU_str.split('\n')
        avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
        avail_GPU_ratio = avail_GPU / total_GPU
        sel_GPU = torch.tensor(avail_GPU_ratio).index_select(0, torch.tensor(self.device_list))
        return self.device_list[np.argmax(sel_GPU)], (sel_GPU.max()>space_reserve)
    
    def rest_gpumem(self, device_id):
        avail_GPU_str = subprocess.getoutput("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free | grep -o '[0-9]\+'")
        avail_GPU = avail_GPU_str.split('\n')
        avail_GPU = np.array([int(device_i) for device_i in avail_GPU])
        return avail_GPU[device_id]
        

    def parallel_run(self, data_x):   
        nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
        res_l = []
        # freeze_support()
        pool = Pool(self.parallel_threads)
        self.total_batch = int(data_x.size()[0]/self.batch_size)
        taskList = list(data_x.chunk(self.total_batch, dim=0))
        task_cnt = 0
        print("there are {} task in this dataset".format(len(taskList)))
        for task in taskList:
            device_id, space_enough = self.avaliable_GPU()
            if(not space_enough):
                torch.cuda.empty_cache()
                time.sleep(0.5)
                device_id, space_enough = self.avaliable_GPU()
            task_cnt += 1
            res = pool.apply_async(func=self.run, args=(task, task_cnt, device_id))
            res_l.append(res)
        pool.close()
        pool.join()
        for res in res_l:
            res.get()
        return task_cnt
        
    def run(self, data_x, batch_id, device_id):
        print("device: {} shot, memory remains: {}MiB".format(device_id, self.rest_gpumem(device_id)))
        device = torch.device(f'cuda:{device_id}')
    
        quantizer = EPQ(
            data_x.to(device),
            block_size=self.block_size,
            n_centroids=self.n_centroids,
            n_iter=self.n_iter,
            eps=self.eps,
            try_cluster=self.try_cluster,
            simplify=self.reduced_PQ,
        )
        
        print(f"Start of Product Quant for batch {batch_id}/{self.total_batch}, \
                  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
        start_time = time.time()
        quantizer.quantize()
        end_time = time.time()
        print(f"End of Product Quant for batch {batch_id}/{self.total_batch}, \
                  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}")
        time_elapse = int(end_time - start_time)
        self.time_record.append(time_elapse)
        
        torch.save(quantizer.centroids, osp.join(self.path, f"batch_{batch_id}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_centroids.pth"))
        torch.save(quantizer.assignments, osp.join(self.path, f"batch_{batch_id}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_assignments.pth"))
        
        del quantizer
        del data_x
        gc.collect()
        torch.cuda.empty_cache()
        return 0
        
    def mergePQ(self, num_batch):
        self.centroids = torch.Tensor()
        self.assignments = torch.Tensor()
        for i in range(num_batch):
            print(f"Merging for {i+1}-th batch...\n")
            centroids = torch.load(osp.join(self.path, f"batch_{i+1}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_centroids.pth"), map_location=self.centroids.device)
            assignments = torch.load(osp.join(self.path, f"batch_{i+1}_blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_assignments.pth"), map_location=self.assignments.device)
            if self.assignments.nelement() == 0:
                self.assignments = assignments.contiguous()
            else:
                reorder_ass = assignments.contiguous() + self.centroids.size()[0]
                self.assignments = torch.cat([self.assignments, reorder_ass], dim=0)
            
            if self.centroids.nelement() == 0:
                self.centroids = centroids.contiguous()
            else:
                self.centroids = torch.cat([self.centroids, centroids.contiguous()], dim=0)
        torch.save(self.centroids, os.path.join(self.path, f"blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_centroids.pth"))
        torch.save(self.assignments, os.path.join(self.path, f"blksize_{self.block_size}_batchsiz_{self.batch_size}_ncents_{self.n_centroids}_assignments.pth"))


if __name__ == '__main__':
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)  #DEBUG < INFO < WARNING < ERROR < CRITICAL
    multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_start_method('forkserver', force=True)
    args = argParse()
    datasets = args.dataset
    data, dataset = dataProcess(datasets)
    reduce = True if datasets in ['Cora', 'Citeseer', 'Pubmed'] else False
    device_list = args.gpu_list
    pr = PR("", "", n_centroids=args.ncents, block_size=args.block_size, parallel_threads=args.pr_thread, 
            path=args.path, device_list=device_list, batch_size=args.batch_size, reduced_PQ=reduce,
            n_iter=args.n_iter, try_cluster=args.try_cluster
            )
    num_batch = pr.parallel_run(data.x)
    print(f"Total time for EPQ: {pr.time_record}s, num_batch:{num_batch}, average time for each batch is:{sum(pr.time_record)/num_batch}")
    pr.mergePQ(num_batch)


