# BLISS (BanditLadies)

Implementation for *"BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks"* paper.

## Abstract
Graph neural networks (GNNs) have become powerful tools for learning representations of graph-structured data. However, training GNNs on large graphs can be computationally expensive, especially when considering all neighbor nodes for each node in the graph. To address the memory and computational bottlenecks encountered when training GNNs on large-scale graphs, we introduce BLISS, a Bandit Layer Importance Sampling Strategy. This approach uses multi-armed bandits to dynamically select the most informative nodes within each layer, striking an optimal balance between exploration and exploitation to ensure both coverage and efficient message passing. Unlike existing sampling methods, BLISS adapts to the evolving graph structure and node importance, leading to more informed node selection and better performance. It demonstrates versatility by effectively working with both Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs), with the bandit framework capable of adapting the selection policy to the specific message aggregation and attention mechanisms of each model. Comprehensive experiments on benchmark datasets show that BLISS maintains or even exceeds the accuracy of full-batch GNN training.

## Installation
```sh
$ pip install torch torchvision torchaudio
$ pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
$ pip install -r requirements.txt
```
or check latest version of [torch](https://www.dgl.ai/pages/start.html) and the combatible version of [dgl](https://www.dgl.ai/pages/start.html) for it. The used version in this project torch\==2.7.0 and dgl\==2.2.1

## How to Run
```sh
$ python train_lightning.py --num-steps 1000 --dataset pubmed --batch-size 32 --num-layers 3 --fan-out 512,256,128 --lr 0.002 --residual --k-runs 5 --sampler poisson-bandit --model sage
```
For reproducibility, from table 3 in the paper:
| Dataset  | Batch Size | Fanouts           | Steps  |
|----------|------------|-------------------|--------|
| Citeseer | 32         | [512, 256, 128]   | 1000   |
| Cora     | 32         | [512, 256, 128]   | 1000   |
| Flickr   | 256        | [4096, 2048, 1024]| 1000   |
| Pubmed   | 32         | [512, 256, 128]   | 1000   |
| Reddit   | 256        | [4096, 2048, 1024]| 3000   |
| Yelp     | 256        | [4096, 2048, 1024]| 10000  |
