# Following from author code cpython version: https://github.com/xavierzw/ogb-geniepath-bs/blob/master/cython_sampler/cython_sampler.pyx
import numpy as np
import math


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

    def update_sample_weights(self, att_map, p, num_data, src_list, dst_list, att_list, neighbor_limit, delta):
      i = 0
      src = 0
      dst = 0
      degree = 0
      idx = 0
      att_val = 0
      reward = 0
      while i < num_data:
        if i % self.num_proc != p:
          i += 1
          continue
        src = src_list[i]
        dst = dst_list[i]
        degree = self.degree[src]
        if degree <= neighbor_limit:
          i += 1
          continue
        delta = delta/degree**2
        idx = self.sample_index[src][dst]
        att_val = att_list[att_map[src][dst]]
        reward = delta*att_val**2/self.sample_probs[src][dst]**2
        if reward > self.max_reward:
          reward = self.max_reward
        self.sample_weights[src][idx] *= math.exp(reward)
        i += 1

    def update_sample_probs(self, p, num_data, src_list, eta):
      i = 0
      idx = 0
      dst = 0
      degree = 0
      unifom_prob = 0
      src = 0
      while i < num_data:
        if i % self.num_proc != p:
          i += 1
          continue
        src = src_list[i]
        degree = self.degree[src]
        if degree <= self.neighbor_limit:
          i += 1
          continue
        weight_sum = sum(self.sample_weights[src])
        unifom_prob = 1./degree

        idx = 0
        while idx < degree:
          dst = self.adj[src][idx]
          self.sample_probs[src][idx] = (1-eta)*self.sample_weights[src][idx] / weight_sum+ eta*unifom_prob
          idx += 1

        i += 1

    def update(self, src_list, dst_list, att_list):
      """Using exp3"""
      
      num_data = len(src_list)
      att_map = None
      i = 0
      while i < num_data:
        att_map[src_list[i]][dst_list[i]] = 1
        i + 1

      p = 0
      neighbor_limit = self.neighbor_limit
      delta = self.delta
      for p in range(self.num_proc):
        self.update_sample_weights(att_map, p. num_data, src_list, dst_list, att_list, neighbor_limit, delta)

      eta = self.data
      src_set = list(set(src_list))
      num_data = len(src_set)
      for p in range(self.num_proc):
        self.update_sample_probs(p, num_data, src_set, eta)



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

    def sample_graph():
      pass