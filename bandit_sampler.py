# Following from author code cpython version: https://github.com/xavierzw/ogb-geniepath-bs/blob/master/cython_sampler/cython_sampler.pyx
import numpy as np


class BanditSampler:
  def __init__(self, neighbor_limit, max_reward, sample_weight, sample_probs, sample_index, adj, sample_set):
    self.neighbor_limit = neighbor_limit
    self.sample_weight = sample_weight
    self.sample_probs = sample_probs
    self.max_reward = max_reward
    self.num_node = adj.shape[0]
    self.degree = np.zeros((self.num_node))
    for src in range(self.num_node):
      if len(adj[src].rows[0]) == 0:
        self.degree[src] = 0
        continue
      dst_list = np.array(adj[src].row[0], dtype=np.int32)
      self.__c_init__(src, dst_list)

    def __c_init__(self, src, dst_list):
      dst_vec = dst_list.to_list()
      degree = len(dst_vec)
      self.degree[src] = degree
      if degree == 0:
        return
      self.sample_weight[src].resize(degree)
      self.sample_probs[src].resize(degree)
      self.adj[src].rezise(degree)
      idx = 0
      while idx < degree:
        dst = dst_vec[idx]
        self.adj[src][idx] = src
        self.sample_weight[src][idx] = 1.
        self.sample_probs[src][idx] = 1./degree
        self.sample_index[src][dst] = idx
        idx += 1

      
    def get_degree(self, src):
      return self.degree[src]
    
    def get_sample_probs(self, src, dst):
      idx = self.sample_index[src][dst]
      return self.sample_probs[idx]

    def get_sample_probs_list(self, src):
      return self.sample_probs[src]

    def get_sample_weights(self, src):
      return self.sample_weight[src]

    def update(self, src_list, dst_list, att_list):
      """Using exp3"""
      pass

    def sample(self, node, sample_size):
      degree = self.degree[node]
      neighbors = []
      if sample_size >= degree:
        return []
      else:
        probs = [x for x in self.sample_probs[node][:degree]]
        print(probs)
        neighbors = np.random.choice(probs, sample_size, p=probs, replace=False)
      return neighbors