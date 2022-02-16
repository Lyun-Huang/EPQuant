# EPQuant
This repository contains the implementation of our paper: EPQuant: An Efficient Quantization Methodology for Graph Neural Networks with Enhanced Product Quantization.

## Dependencies
Our code works with Python 3.8 and newst. To run the code, you must have the following packages installed:
- [Pytorch](http://pytorch.org/) (version=1.9.0)
- [PyG](https://github.com/rusty1s/pytorch_geometric) (version=2.0.0)

## Training and evaluate
To train the model with EPQuant, the dataset(s) need to be quantized with EPQ first. You can use the command we provide if you use one of datasets in Cora, Citeseer, PubMed, Reddit, and Amazon2M/ogbn-products.
A example command:
```
python epq_alone.py --dataset='Cora' --block_size=4 --try_cluster=15 --n_iter=15
```
If you need to quantize other datasets, a few modification to this script can be achieved.

To train and evaluate the quantized model, run the following command (for GCN and GAT):
```
python epq_sparse.py --model='GCN' --dataset='Cora' --block_size=4 --hidden=16 --epoch=600 --nruns=10 --pqnt --act_qnt --wt_qnt --wf
```
or the follwing command (for GraphSAGE):
```
python epq_dense.py --model='GS-mean' --dataset='Reddit' --block_size=32  --epoch=6
```

## Run without downloading datasets
We provide several qauntized datasets which you can find in `pq_data`. You can train and evaluate models without downloading these datasets, use the following command:
```
python epq_sparse.py --fast --block_size=22 --model='GCN' --dataset='Cora'
```

## Pre-trained model
We provide some pre-trained model in this repo to reproduct the results in our paper, which you can find in `pretrained`. Run this command:
```
python epq_sparse.py --fast --block_size=22 --model='GCN' --dataset='Citeseer' --pretrained
```

Everytime you run `epq_sparse.py` or `epq_dense.py`, you can see the results concluded as follows:   

| data type   | size before quant(MB) | size after quant(MB) | compression ratio |  
| :---------: | :-------------------: | :------------------: | :--------------:  |  
| input_layer1 |        46.9966        |        0.1425        |      329.7734     |
| input_layer2 |         0.2031        |        0.0508        |       4.0000      |
|  model_size  |         0.2264        |        0.0566        |       4.0000      |
| overall_size |        47.4261        |        0.2499        |      189.8009     |

| accuracy | mean(%) | lower boundary(%) | upper boundary(%) |  
| :------: | :-----: | :---------------: | :---------------: |
| accuracy |  0.6830 |       0.0000      |       0.0000      | 
