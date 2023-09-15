# source: https://github.com/dmlc/dgl/blob/d024c1547279d837350bd356df9b693d44bb49e2/examples/pytorch/labor/ladies_sampler.py#L25
import dgl.function as fn
import dgl
import torch

def find_indices_in(a, b):
    # sort b in ascending order by value, and indices of the elements in the original input tensor.
    b_sorted, indices = torch.sort(b)
    sorted_indices = torch.searchsorted(b_sorted, a)
    sorted_indices[sorted_indices >= indices.shape[0]] = 0
    return indices[sorted_indices]

def union(*arrays):
    # concat the passed arrays vertically and return the unique values only as 1-dim tensor
    return torch.unique(torch.cat(arrays))

def normalized_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
        g.apply_edges(lambda edges: {'w': 1 / edges.dst['v']})
        return g.edata['w']

class BanditSampler(dgl.dataloading.BlockSampler): # consider to use unbiased node embedding and edge_weights
    def __init__(self, nodes_per_layer, importance_sampling=True, weight='w', out_weight='edge_weights',
                 node_embedding='nfeat', node_prob='node_prob', replace=False, eta=0.4, T=40,
                 allow_zero_in_degree=False, model='gat'):
        super().__init__()
        self.nodes_per_layer = nodes_per_layer
        self.importance_sampling = importance_sampling
        self.edge_weight = weight
        self.output_weight = out_weight
        self.node_prob = node_prob
        self.node_embedding = node_embedding
        self.replace = replace
        self.eta = eta
        self.T = T
        self.exp3_weights = None # I think with each blocks we need to have a separate exp3_weights
        self.allow_zero_in_degree = allow_zero_in_degree
        self.model = model
    
    def compute_prob(self, g, seed_nodes, weight):
        """
        Args:
            g (DGLGraph): the whole graph
            seed_nodes (Tensor): the output nodes for the current layer
            weight (Tensor): the weight of the edges
        
        Returns:
            Tensor: the unnormalized probability of the candidate nodes
            DGLBlock: subgraph containing all the edges from the candidate nodes to the output nodes.
        """
        # create new subgraph using the incoming edges of the given nodes
        insg = dgl.in_subgraph(g, seed_nodes)
        # eliminate the isolated nodes across graph
        insg = dgl.compact_graphs(insg, seed_nodes)
        if self.importance_sampling:
            out_frontier = dgl.reverse(insg, copy_edata=True)
            weight = weight# [out_frontier.edata[dgl.EID].long()] 
            weight_sum = dgl.ops.copy_e_sum(out_frontier, weight)
            weight_div_sum = dgl.ops.e_div_v(out_frontier, weight, weight_sum)
            prob = dgl.ops.copy_e_sum(out_frontier, weight_div_sum ** 2)
            prob = torch.sqrt(prob)
        else:
            prob = torch.ones(insg.num_nodes())
            prob[insg.out_degrees() == 0] = 0
        return prob, insg

    def select_neighbors(self, prob, num):
        """
        Args:
            prob (Tensor): unnormalized probability of each candidate node
            num (int): number of neighbors to sample
        
        Returns:
            Tensor: the set of input nodes in terms of their indices in @cand_nodes, and also the indices of
                    seed nodes in the selected nodes.
        """
        # The returned nodes should be a union of seed_nodes plus @num nodes from cand_nodes.
        # Because compute_prob returns a compacted subgraph and a list of probabilities,
        # we need to find the corresponding local IDs of the resulting union in the subgraph
        # so that we can compute the edge weights of the block.
        # This is why we need a find_indices_in() function.
        
        # sample K nodes given the node probabilities,
        # where K is the passed num or the number of nodes in prob if num is larger than available nodes
        neighbor_nodes_idx = torch.multinomial(prob, min(num, prob.shape[0]), replacement=self.replace)
        return neighbor_nodes_idx

    def calculate_alpha(self, mfg):
        if self.model == 'sage':
            # alpha
            alpha = mfg.edata[self.edge_weight]
        elif self.model == 'gat':
            q_ij = mfg.edata['q_ij']
            attention = mfg.edata['a_ij']

            q_ij_sum = dgl.ops.copy_e_sum(mfg, q_ij)
            attention_sum = dgl.ops.copy_e_sum(mfg, attention)
            attention_div_attention_sum = dgl.ops.e_div_v(mfg, attention, attention_sum)
            attention_div_attention_sum = torch.nan_to_num(attention_div_attention_sum)
            alpha = dgl.ops.e_dot_v(mfg, attention_div_attention_sum, q_ij_sum)
        return alpha

    def calculate_rewards(self, idx, mfg, alpha, epsilon=1e-5):
        """
        Calculate the reward for each edge in the @mfgs.

        Args:
            mfgs (DGLBlock): the blocks (in top-down format) to compute rewards for
            epsilon (float): add small value to the prob to avoid dividing by zero

        Returns:
            DGLBlock: the mfgs after adding reward for each edge to each of mfgs 
        """
        # k is number of nodes in the subgraph (sample size or number of arms)
        # calculate ||h_j(t)|| (node embedding norm)
        # node prob
        # calculate rewards (alpha**2 * ||h_j(t)||**2) / (k * q**2)
        # store rewards inside the block data
        k = self.nodes_per_layer[idx]
        h_j_norm = mfg.srcdata['embed_norm']
        q = mfg.srcdata[self.node_prob]
        rewards = dgl.ops.e_mul_u(mfg, alpha**2, (h_j_norm ** 2) / (k * (q ** 2)))
        mfg.edata['rewards'] = rewards
        
    
    def update_exp3_weights(self, idx, mfg, g):
        """
        Update the exp3 weights of each edge being selected using the rewards obtained
        from the previous selection using exp3 algo.

        Args:
            mfgs (DGLBlock): the blocks (in top-down format) to compute exp3 edge weight for
        """
        # T = self.T, Total number of iterations.
        # eta = self.eta, Learning rate for updating the probability.
        # Number of nodes to select in each iteration (sample size or number of arms).
        # Calculate the delta value for exp3
        # delta: sqrt((1 - eta) * eta^4 * k^5 * ln(n/k) / (T*n^4))
        k = self.nodes_per_layer[idx]
        n = g.in_degrees()[mfg.srcdata[dgl.NID].long()]
        delta = torch.sqrt((1 - self.eta) * self.eta**4 * k**5 * torch.log(n/k) / (self.T * n**4))
        delta = torch.nan_to_num(delta)
        rewards = mfg.edata['rewards'].clone().detach()
    
        # Unnormalized probability distribution of edges in the current subgraph.
        prob = mfg.edata['q_ij'].clone().detach()

        # Compute rewards_hat by dividing rewards by edge probabilities
        rewards_hat = rewards**2 / prob**2

        # Compute exponential weight for edges
        # delta_reward = dgl.ops.e_mul_u(mfg, rewards_hat, delta / n)
        delta_reward = dgl.ops.e_mul_u(mfg, rewards_hat, delta)
        delta_reward[delta_reward > 1] = 1
        exp_rewards = torch.exp(delta_reward)
        
        # update weights
        self.exp3_weights[idx][mfg.edata[dgl.EID].long()] *= exp_rewards

    def exp3_probabilities(self, idx, g, seed_nodes):
        """
        Update the exp3 probability of each edge being selected using the updated exp3 weights using exp3 algo.

        Args:
            idx (int): block index
            g (DGLGraph): the original graph
            seed_nodes (Tensor): node ids to calculate edge prob for

        Returns:
            Tensor: the normalized prob of each edge (edge_prob)
        """
        # eta = self.eta, Learning rate for updating the probability.
        edges_ids = g.in_edges(seed_nodes, form='eid')
        exp_weights = self.exp3_weights[idx] #[mfg.edata[dgl.EID].long()]
        exp3_weights_sum = dgl.ops.copy_e_sum(g, exp_weights)
        exp_weights_divided = dgl.ops.e_div_v(g, exp_weights, exp3_weights_sum) # was E_DIV_U
        k = g.in_degrees()
        edge_prob = dgl.ops.u_add_e(g, (self.eta / k), (1 - self.eta) * (exp_weights_divided))

        return edge_prob[edges_ids.long()]
    
    def exp3(self, mfgs, g):
        """_summary_

        Args:
            mfgs (DGLBlock): _description_
            g (DGLGraph): _description_
        """        
        # loop over all blocks (layers)
        for idx, mfg in enumerate(mfgs):
            alpha = self.calculate_alpha(mfg)

            self.calculate_rewards(idx, mfg, alpha)
            self.update_exp3_weights(idx, mfg, g)

    def generate_block(self, insg, neighbor_nodes_idx, seed_nodes, P_sg, W_sg):
        """
        Args:
            insg : the subgraph yielded by compute_prob()
            neighbor_nodes_idx (Tensor): the sampled nodes from the subgraph @insg, yielded by select_neighbors()
            seed_nodes_local_idx (Tensor): the indices of seed nodes in the selected neighbor nodes, also yielded
                                           by select_neighbors()
            P_sg (Tensor): unnormalized probability of each node being sampled, yielded by compute_prob()
            W_sg (Tensor): edge weights of @insg
        
        Returns:
            DGLBlock: the block.
        """
        # find indices of seed_nodes in the subgraph nodes
        seed_nodes_idx = find_indices_in(seed_nodes, insg.ndata[dgl.NID])
        u_nodes = union(neighbor_nodes_idx, seed_nodes_idx)
        sg = insg.subgraph(u_nodes.type(insg.idtype))
        u, v = sg.edges()
        lu = sg.ndata[dgl.NID][u.long()]
        s = find_indices_in(lu, neighbor_nodes_idx)
        
        # sample subgraph using the given edges (or boolean mask of the available nodes)
        eg = dgl.edge_subgraph(sg, lu == neighbor_nodes_idx[s], relabel_nodes=False)
        eg.ndata[dgl.NID] = sg.ndata[dgl.NID][:eg.num_nodes()]
        eg.edata[dgl.EID] = sg.edata[dgl.EID][eg.edata[dgl.EID].long()]
        sg = eg
        nids = insg.ndata[dgl.NID][sg.ndata[dgl.NID].long()]

        P = P_sg[u_nodes.long()]
        W = W_sg[sg.edata[dgl.EID].long()]
        
        W_tilde = dgl.ops.e_div_u(sg, W, P)
        W_tilde_sum = dgl.ops.copy_e_sum(sg, W_tilde)
        d = sg.in_degrees()
        W_tilde = dgl.ops.e_mul_v(sg, W_tilde, d / W_tilde_sum)

        # W_tilde = dgl.ops.e_mul_v(sg, W, d[u_nodes].float())

        block = dgl.to_block(sg, seed_nodes_idx.type(sg.idtype))
        block.edata[self.output_weight] = W_tilde
        block.edata['q_ij'] = W
        block.srcdata[self.node_prob] = P

        # get node ID mapping for source nodes
        block.srcdata[dgl.NID] = nids[block.srcdata[dgl.NID].long()]
        block.dstdata[dgl.NID] = nids[block.dstdata[dgl.NID].long()]
        sg_eids = insg.edata[dgl.EID][sg.edata[dgl.EID].long()]
        block.edata[dgl.EID] = sg_eids[block.edata[dgl.EID].long()]

        return block
    
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        if self.exp3_weights == None:
          self.exp3_weights = torch.ones(len(self.nodes_per_layer), g.num_edges()).to(g.device)
        seed_nodes = torch.tensor(seed_nodes)
        output_nodes = seed_nodes
        blocks = []
        for block_id in reversed(range(len(self.nodes_per_layer))):
            num_nodes_to_sample = self.nodes_per_layer[block_id]
            # calc exp3_prob, 1 / N_i
            exp3_prob = self.exp3_probabilities(block_id, g, seed_nodes)

            # prob, insg = self.compute_prob(g, seed_nodes, exp3_prob)
            # W = exp3_prob #[insg.edata[dgl.EID].long()]
            
            W = g.edata[self.edge_weight]
            prob, insg = self.compute_prob(g, seed_nodes, W)
            chosen_nodes = self.select_neighbors(prob, num_nodes_to_sample)
            block = self.generate_block(insg, chosen_nodes.type(g.idtype), seed_nodes.type(g.idtype), prob, W)
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks
