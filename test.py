import numpy as np
import torch
import dgl
import networkx as nx
from dgl_bandit_sampler import BanditSampler
from ladies import LadiesSampler

# Test on Toy graph
g = dgl.graph(([2, 3, 3, 4], [0, 0, 1, 1] ), num_nodes=5, col_sorted=True)
c = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]] , dtype= torch.float)

# Assign a 3-dimensional node feature vector for each node.
g.ndata['nfeat'] = torch.tensor([[0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]] , dtype= torch.float)
# Assign a 1-dimensional edge feature vector for each edge.
g.edata['w'] = torch.FloatTensor([0.5, 0.5, 0.2, 0.8])

seed_nodes = [0, 1]
weight = g.edata['w']

insg = dgl.in_subgraph(g, seed_nodes)
insg = dgl.compact_graphs(insg, seed_nodes)
out_frontier = dgl.reverse(insg, copy_edata=True)
weight = weight[out_frontier.edata[dgl.EID].long()]
prob = dgl.ops.copy_e_sum(out_frontier, weight ** 2)

print("Probcability of nodes: ", prob)

# prob = torch.tensor([0, 0, 0.5, 0.5, 0])
# torch.multinomial(prob, min(3, len(prob)), replacement=False)


bandit = BanditSampler([2], weight='w')
seed_nodes, output_nodes, blocks = bandit.sample_blocks(g, [0, 1])
print("Bandit Seed nodes: ", seed_nodes)
print("Bandit Output nodes: ", output_nodes)

ladies = LadiesSampler([2], weight='w')
seed_nodes, output_nodes, blocks = ladies.sample_blocks(g, [0, 1])
print("Ladies Seed nodes: ", seed_nodes)
print("Ladies Output nodes: ", output_nodes)