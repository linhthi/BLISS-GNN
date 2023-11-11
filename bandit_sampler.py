# source: https://github.com/dmlc/dgl/blob/d024c1547279d837350bd356df9b693d44bb49e2/examples/pytorch/labor/ladies_sampler.py#L25
import dgl
import torch

def find_indices_in(a, b):
    # sort b in ascending order by value, and indices of the elements in the original input tensor.
    b_sorted, indices = torch.sort(b)
    # find indices of a where elements should be inserted in b to maintain order.
    # e.g., b = [1, 2, 3, 4, 5], a = [-10, 10, 2, 3] >>> [0, 5, 1, 2]
    sorted_indices = torch.searchsorted(b_sorted, a)
    # remove sorted_indices of value >= 0-dimension (rows) of the indices
    sorted_indices[sorted_indices >= indices.shape[0]] = 0
    # return the matched indices of a in b
    return indices[sorted_indices]

def union(*arrays):
    # concat the passed arrays vertically and return the unique values only as 1-dim tensor
    return torch.unique(torch.cat(arrays))

def normalized_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = 'W'
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g.update_all(dgl.function.copy_e(weight, weight), dgl.function.sum(weight, 'v'))
        g.apply_edges(lambda edges: {'w': 1 / edges.dst['v']})
        return g.edata['w'] * g.edata[weight]

class BanditLadiesSampler(dgl.dataloading.BlockSampler): # consider to use unbiased node embedding and edge_weights
    def __init__(self, nodes_per_layer, importance_sampling=True, weight='w', out_weight='edge_weights',
                 node_embedding='nfeat', node_prob='node_prob', replace=False, eta=0.4, num_steps=5000,
                 allow_zero_in_degree=False, model='sage'):
        super().__init__()
        self.nodes_per_layer = nodes_per_layer
        self.importance_sampling = importance_sampling
        self.edge_weight = weight
        self.output_weight = out_weight
        self.node_prob = node_prob
        self.node_embedding = node_embedding
        self.replace = replace
        self.eta = eta
        self.T = num_steps
        self.exp3_weights = None
        self.allow_zero_in_degree = allow_zero_in_degree
        self.model = model
        # self.converge = [[] for _ in range(len(self.nodes_per_layer))]
    
    def compute_prob(self, insg, edge_prob, num):
        r"""
        Args:
            insg (DGLGraph): subgraph of the seed nodes graph
            edge_prob (Tensor): the edge probability
            num : number of neighbors to sample

        STEP_02    

        Equation:
        q_j = \sqrt{\sum_{i} (\frac{q_{ij}}{\sum_{k\in\mathcal{N}_i}q_{ik}})^2}
            - q_j is the prob of node being sampled
            - q_ij is the edge prob, it is an estimate of alpha_ij * ||h_j||
        
        Returns:
            Tensor: the unnormalized probability of the candidate nodes
            DGLBlock: subgraph containing all the edges from the candidate nodes to the output nodes.
        """
        if self.importance_sampling:
          # \sum_{k\in\mathcal{N}_i}q_{ik}, calc sum of q_ij over j (to get i sum)
          edge_prob_sum = dgl.ops.copy_e_sum(insg, edge_prob)
          # reverse the edges (top-down)
          out_frontier = dgl.reverse(insg, copy_edata=True)
          # \frac{q_{ij}}{weight_sum}
          edge_prob_div_sum = dgl.ops.e_div_u(out_frontier, edge_prob, edge_prob_sum)          
          # \sum_{i} (weight_div_sum)^2, prob for each node
          prob = dgl.ops.copy_e_sum(out_frontier, edge_prob_div_sum ** 2)
          # \sqrt{prob}, take the square root to follow the importance sampling equation
          prob = torch.sqrt(prob)
        else:
          # prob for choosing any neighbor is 1
          prob = torch.ones(insg.num_nodes())
          # set the non neighboring nodes to 0
          prob[insg.out_degrees() == 0] = 0
        return prob
    
    def select_neighbors(self, prob, num):
        """
        seed_nodes : output nodes
        cand_nodes : candidate nodes.  Must contain all output nodes in @seed_nodes
        prob : unnormalized probability of each candidate node
        num : number of neighbors to sample
        return : the set of input nodes in terms of their indices in @cand_nodes, and also the indices of
                 seed nodes in the selected nodes.
        """
        # The returned nodes should be a union of seed_nodes plus @num nodes from cand_nodes.
        # Because compute_prob returns a compacted subgraph and a list of probabilities,
        # we need to find the corresponding local IDs of the resulting union in the subgraph
        # so that we can compute the edge weights of the block.
        # This is why we need a find_indices_in() function.
        neighbor_nodes_idx = torch.multinomial(prob, min(num, prob.shape[0]), replacement=self.replace)
        return neighbor_nodes_idx

    def exp3_probabilities(self, idx, g, seed_nodes):
        r"""
        Update the exp3 probability of each edge being selected using the updated exp3 weights using exp3 algorthim.

        STEP_01

        Equation:
        q_{ij} = (1 - \eta) \frac{w_{ij}}{\sum_{j} w_{ij}} + \frac{\eta}{n_i}
            - q_ij is the probability of selecting edge ij
            - η (eta) is the learning rate
            - n_i is the number of edges incoming to the seed node (degree) or the number neighbors of the node i
            - w_ij is the exp3 weight of edge ij

        Args:
            idx (int): block index
            g (DGLGraph): the original graph
            seed_nodes (Tensor): node ids to calculate edge prob for

        Returns:
            Tensor, DGLGraph: the normalized prob of each edge (edge_prob), and subgraph
        """
        # create new subgraph using the incoming edges of the given nodes
        insg = dgl.in_subgraph(g, seed_nodes)
        # eliminate the isolated nodes across graph
        insg = dgl.compact_graphs(insg, seed_nodes)
        # update weights (w_ij)
        exp_weights = self.exp3_weights[idx][insg.edata[dgl.EID].long()]
        # \sum_{j} w_{ij}, sum of weights per node
        # print('exp_weights', exp_weights, exp_weights.shape)
        exp3_weights_sum = dgl.ops.copy_e_sum(insg, exp_weights)
        # print('exp3_weights_sum', exp3_weights_sum, exp3_weights_sum.shape)
        # \frac{w_{ij}}{exp3_weights_sum}, divide exp_weights over exp_weights_sum
        exp_weights_divided = dgl.ops.e_div_v(insg, exp_weights, exp3_weights_sum)
        # print('exp_weights_divided', exp_weights_divided, exp_weights_divided.shape)
        # number of edges incoming to the seed node (degree)
        n_i = g.in_degrees(insg.srcdata[dgl.NID])
        # print('\nni', n_i, n_i.shape)
        # print('\nni_2', g.in_degrees(insg.dstdata[dgl.NID]), g.in_degrees(insg.dstdata[dgl.NID]).shape)
        # print('\nni_3', g.in_degrees()[insg.dstdata[dgl.NID].long()], g.in_degrees()[insg.dstdata[dgl.NID].long()].shape)
        # print('\nni_2', (g.in_degrees()[insg.srcdata[dgl.NID].long()] == n_i).all())

        # update edge prob 
        # (1 - \eta) {exp_weights_divided} + \frac{\eta}{k}
        edge_prob = dgl.ops.v_add_e(insg, (self.eta / n_i), (1 - self.eta) * (exp_weights_divided)) 
        # print('edge_prob', edge_prob, edge_prob.shape)
        return edge_prob, insg
    
    def calculate_alpha(self, mfg):
        '''
        STEP_05

        alpha is the original graph weights (constant for GraphSAGE)
        '''
        if self.model == 'sage':
            # alpha
            alpha = mfg.edata[self.edge_weight]
        elif self.model == 'gat':
            # # print('len nodes', mfg.srcdata['labels'].shape)
            q_ij = mfg.edata['q_ij']
            # # print('q_ij', q_ij, q_ij.shape)
            attention = mfg.edata['a_ij']
            # # print('attention', attention, attention.shape)
            q_ij_sum = dgl.ops.copy_e_sum(mfg, q_ij)
            # # print('q_ij_sum', q_ij_sum, q_ij_sum.shape)
            attention_sum = dgl.ops.copy_e_sum(mfg, attention)
            # # print('attention_sum', attention_sum, attention_sum.shape)
            attention_div_attention_sum = dgl.ops.e_div_v(mfg, attention, attention_sum) # [!] _u?
            # print('attention_div_attention_sum', attention_div_attention_sum, attention_div_attention_sum.shape)
            attention_div_attention_sum = torch.nan_to_num(attention_div_attention_sum)
            # print('attention_div_attention_sum', attention_div_attention_sum, attention_div_attention_sum.shape)
            alpha = dgl.ops.e_dot_v(mfg, attention_div_attention_sum, q_ij_sum)
            # # print('alpha', alpha, alpha.shape)
        return alpha

    def calculate_rewards(self, idx, mfg, g, alpha):
        r"""
        Calculate the reward for each edge in the @mfg.

        STEP_06

        # Equation:
        r_{ij} = \frac{\alpha_{ij}^2}{k\cdot q_j^2} \|h_j\|_2^2
            - r_ij is the reward for the edge ij
            - alpha original graph edge weights.
            - q_j node prob.
            - h_j node embedding.

        Args:
            mfgs (DGLBlock): the blocks (in top-down format) to compute rewards for

        Returns:
            DGLBlock: the mfgs after adding reward for each edge to each of mfgs 
        """
        # # n_i is number of nodes in the subgraph (sample size or number of arms)
        # n_i = g.in_degrees()[mfg.dstdata[dgl.NID].long()]

        # Number of nodes to select in each iteration (sample size or number of arms).
        k_i = mfg.in_degrees()[:len(mfg.dstdata[dgl.NID])]
        # print('\nki', k_i, k_i.shape)
        
        # calculate \|h_j\| (node embedding norm)
        h_j_norm = mfg.srcdata['embed_norm']
        # print('h_j_norm.shape', h_j_norm.shape)
        # node prob
        # q = mfg.srcdata[self.node_prob]
        q_ij = mfg.edata['q_ij']
        # \frac{\alpha_{ij}^2}{k_i}
        alpha_div_k_i = dgl.ops.e_div_v(mfg, alpha**2, k_i)
        alpha_div_k_i = torch.nan_to_num(alpha_div_k_i, posinf=0)
        # print()
        # print('alpha_div_k_i=========', alpha_div_k_i.min().tolist(), alpha_div_k_i.max().tolist())

        # \frac{\|h_j\|_2^2}{q_{ij}^2}
        h_j_norm_div_q_j = dgl.ops.u_div_e(mfg, h_j_norm ** 2, q_ij ** 2)
        # print('h_j_norm_div_q_j=========', h_j_norm_div_q_j.min().tolist(), h_j_norm_div_q_j.max().tolist())

        # \frac{\alpha_{ij}^2}{k\cdot q_j^2} \|h_j\|_2^2, calculate rewards
        rewards = alpha_div_k_i * h_j_norm_div_q_j
        # store rewards inside the block data
        # print('rewards=========', rewards.min(), rewards.max())
        mfg.edata['rewards'] = rewards

    def update_exp3_weights(self, idx, mfg, g):
        r"""
        Update the exp3 weights of each edge being selected using the rewards obtained
        from the previous selection using exp3 algo.

        STEP_07

        Equation:
        w_{ij} = w_{ij} \exp(\frac{\delta r_{ij}}{n_i p_i})
            - w_ij is the exp3 weight of edge ij
            - n_i is the number of neighbors of the node at the end of edge i (degree)
            - r_ij is the reward obtained for selecting edge ij in the previous iteration
            - p_i is the probability of selecting a node i in the previous iteration
            - δ (delta) is the learning rate
        
        \delta = \sqrt{\frac{(1 - \eta) \eta^4 k^5 \ln(\frac{n}{k})}{T n^4}}
            - eta is the learning rate
            - k is the number of nodes to select in each iteration
            - n is the number of nodes in the graph
            - T is the total number of iterations


        Args:
            mfgs (DGLBlock): the blocks (in top-down format) to compute exp3 edge weight for
        """
        # Number of nodes to select in each iteration (sample size or number of arms).
        k_i = mfg.in_degrees()[:len(mfg.dstdata[dgl.NID].long())]
        # Number of nodes in the current subgraph (neigbor of a node or degree)
        n_i = g.in_degrees()[mfg.dstdata[dgl.NID].long()]

        # Calculate the delta value for exp3
        # delta: \sqrt{\frac{(1 - \eta) \eta^4 k^5 \ln(\frac{n}{k})}{T n^4}}
        nom = (1-self.eta)*(self.eta**4)*(k_i**5)*torch.log(n_i/k_i)
        dom = (self.T*n_i**4)
        delta = torch.sqrt(nom/dom)
        delta = torch.nan_to_num(delta)
        # print('delta', delta)

        # delta = self.eta / n_i**2
        # delta = self.eta / 100
        # delta = 0.01

        # The rewards obtained for each node in the previous iteration
        rewards = mfg.edata['rewards'].clone().detach()
        # print()
        # print('+'*50)
        # print('rewards=========', rewards.min().tolist(), rewards.max().tolist())
        # Unnormalized probability distribution of nodes in the current subgraph.
        prob = mfg.srcdata[self.node_prob].clone().detach()
        # \hat{r}_{ij} = \frac{r_{ij}}{p_i}, Compute rewards_hat by dividing rewards by node probabilities
        rewards_hat = dgl.ops.e_div_u(mfg, rewards, prob)
        # print('rewards_hat=========', rewards_hat.min().tolist(), rewards_hat.max().tolist())
        # \frac{\delta \hat{r}_{ij}}{k p_i}, compute delta_reward for edges
        delta_reward = dgl.ops.e_mul_v(mfg, rewards_hat, delta / n_i)
        # print('delta_reward=========', delta_reward.min().tolist(), delta_reward.max().tolist())
        # print('+'*50)
        # print('delta_reward', delta_reward.max())
        # # limit delta_reward to 1
        # delta_reward[delta_reward > 1] = 1
        # \exp(\frac{\delta \hat{r}_{ij}}{k p_i}), take exp of delta_reward
        exp_rewards = torch.exp(delta_reward)
        # update weights
        self.exp3_weights[idx][mfg.edata[dgl.EID].long()] *= exp_rewards
    
    def exp3(self, mfgs, g):
        """
        EXP3 algorhim execution after doing feed-forward, to get the node embeddings
        of the last iteration. 

        Args:
            mfgs (DGLBlock): blocks containing the sub graphs
            g (DGLGraph): original graph
        """
        # loop over all blocks (layers)
        for idx, mfg in enumerate(mfgs):
            alpha = self.calculate_alpha(mfg)
            # calculate rewards
            self.calculate_rewards(idx, mfg, g, alpha)
            # update exp3 weights
            self.update_exp3_weights(idx, mfg, g)

    def generate_block(self, insg, neighbor_nodes_idx, seed_nodes, P_sg, W_sg):
        """
        STEP_04

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
        # union of node idx from both seed_nodes and neighbor_nodes
        u_nodes = union(neighbor_nodes_idx, seed_nodes_idx)
        # sample subgraph for the union nodes
        sg = insg.subgraph(u_nodes.type(insg.idtype))
        # return source (u) and destination (v) nodes
        u, v = sg.edges()
        # sources nodes (actual (original) indices)
        lu = sg.ndata[dgl.NID][u.long()]
        # find matched indices of lu (source nodes) in neighbor_nodes
        s = find_indices_in(lu, neighbor_nodes_idx)
        
        # sample subgraph using the given edges (or boolean mask of the available nodes)
        eg = dgl.edge_subgraph(sg, lu == neighbor_nodes_idx[s], relabel_nodes=False)
        # update the node data with the original node data
        eg.ndata[dgl.NID] = sg.ndata[dgl.NID][:eg.num_nodes()]
        # update the edge data with the original edge data
        eg.edata[dgl.EID] = sg.edata[dgl.EID][eg.edata[dgl.EID].long()]
        # update the first subgraph with the updated edge subgraph
        sg = eg
        # # print('sg', sg)
        # get the original node ids from g
        nids = insg.ndata[dgl.NID][sg.ndata[dgl.NID].long()]

        # get the probability for union nodes from P_sg which is unnormalized probability of the original subgraph.
        P = P_sg[u_nodes.long()]
        # get the edge weight for the edges in the current subgraph from W_sg which is edge weight from the original subgraph.
        W = W_sg[sg.edata[dgl.EID].long()]
        
        # divide W over P
        W_tilde = dgl.ops.e_div_u(sg, W, P)
        # sum of W_tilde per node
        W_tilde_sum = dgl.ops.copy_e_sum(sg, W_tilde)
        # get nodes degrees
        d = sg.in_degrees()
        # multiply W_tilde by d divided by W_tilde_sum
        W_tilde = dgl.ops.e_mul_v(sg, W_tilde, d / W_tilde_sum)
        # print('W_tilde', W_tilde)

        # W_tilde = dgl.ops.e_mul_v(sg, W, d[u_nodes].float())

        block = dgl.to_block(sg, seed_nodes_idx.type(sg.idtype))
        # update the edge data with W_tilde
        block.edata[self.output_weight] = W_tilde
        # add edge prob
        block.edata['q_ij'] = W
        # add node prob
        # print('P', P, P.shape)
        # print()
        block.srcdata[self.node_prob] = P

        # get node ID mapping for source nodes
        block.srcdata[dgl.NID] = nids[block.srcdata[dgl.NID].long()]
        # set node ID mapping for destination nodes
        block.dstdata[dgl.NID] = nids[block.dstdata[dgl.NID].long()]
        # get the original edge ids 
        sg_eids = insg.edata[dgl.EID][sg.edata[dgl.EID].long()]
        # set the original edge ids to the block
        block.edata[dgl.EID] = sg_eids[block.edata[dgl.EID].long()]

        return block
    
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        if self.exp3_weights == None:
          self.exp3_weights = torch.ones(len(self.nodes_per_layer), g.num_edges()).to(g.device)
        
        # convert seed_nodes IDs to tensor
        # seed_nodes = torch.tensor(seed_nodes)
        # copy seed_nodes to output_nodes (seed_nodes will be updated, output_nodes not)
        output_nodes = seed_nodes
        # empty list 
        blocks = []
        # loop on the reverse of block IDs
        # print('+'*30)
        for block_id in reversed(range(len(self.nodes_per_layer))):
            # print('-'*10, f'BLOCK{block_id}', '-'*10)
            # get the number of sample from nodes_per_layer per each block
            num_nodes_to_sample = self.nodes_per_layer[block_id]
            # calc exp3_prob, 1 / N_i
            edge_prob, insg = self.exp3_probabilities(block_id, g, seed_nodes)
            # print('exp3_prob', edge_prob, edge_prob.shape)
            # self.converge[block_id].append(edge_prob.tolist())
            # run compute_prob to get the unnormalized prob and subgraph
            node_prob = self.compute_prob(insg, edge_prob, num_nodes_to_sample)
            # print('node_prob', node_prob, )
            # get the edge prob from the original graph (exp3)
            W = edge_prob
            # sample the best n neighbor nodes from given the probabilities of neighbors (and the current nodes)
            chosen_nodes = self.select_neighbors(node_prob, num_nodes_to_sample)
            # print('chosen_nodes', chosen_nodes, chosen_nodes.shape)
            # generate block for the sampled nodes and the previous nodes
            block = self.generate_block(insg, chosen_nodes.type(g.idtype), seed_nodes.type(g.idtype), node_prob, W)
            # update the seed_nodes with the sampled neighbors nodes to sample another block foe them in the next iteration
            seed_nodes = block.srcdata[dgl.NID]
            # add blocks at the beginning of blocks list (top-down)
            blocks.insert(0, block)
        # print('+'*30)
        return seed_nodes, output_nodes, blocks

class PoissonBanditLadiesSampler(BanditLadiesSampler):
    def __init__(
        self, nodes_per_layer, importance_sampling=True, weight='w', out_weight='edge_weights',
        node_embedding='nfeat', node_prob='node_prob', replace=False, eta=0.4, num_steps=5000,
        allow_zero_in_degree=False, model='sage'
    ):
        super().__init__(
            nodes_per_layer, importance_sampling, weight, out_weight,
            node_embedding, node_prob, replace, eta, num_steps,
            allow_zero_in_degree, model)
        self.eps = 0.9999

    def compute_prob(self, insg, edge_prob, num):
        """
        g : the whole graph
        seed_nodes : the output nodes for the current layer
        weight : the weight of the edges
        return : the unnormalized probability of the candidate nodes, as well as the subgraph
                 containing all the edges from the candidate nodes to the output nodes.
        """
        prob = super().compute_prob(insg, edge_prob, num)

        one = torch.ones_like(prob)
        if prob.shape[0] <= num:
            return one

        c = 1.0
        for i in range(50):
            S = torch.sum(torch.minimum(prob * c, one).to(torch.float64)).item()
            if min(S, num) / max(S, num) >= self.eps:
                break
            else:
                c *= num / S

        return torch.minimum(prob * c, one)

    def select_neighbors(self, prob, num):
        """
        seed_nodes : output nodes
        cand_nodes : candidate nodes.  Must contain all output nodes in @seed_nodes
        prob : unnormalized probability of each candidate node
        num : number of neighbors to sample
        return : the set of input nodes in terms of their indices in @cand_nodes, and also the indices of
                 seed nodes in the selected nodes.
        """
        # The returned nodes should be a union of seed_nodes plus @num nodes from cand_nodes.
        # Because compute_prob returns a compacted subgraph and a list of probabilities,
        # we need to find the corresponding local IDs of the resulting union in the subgraph
        # so that we can compute the edge weights of the block.
        # This is why we need a find_indices_in() function.
        neighbor_nodes_idx = torch.arange(prob.shape[0], device=prob.device)[
            torch.bernoulli(prob) == 1
        ]
        return neighbor_nodes_idx