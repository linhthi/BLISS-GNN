# source: https://github.com/dmlc/dgl/blob/d024c1547279d837350bd356df9b693d44bb49e2/examples/pytorch/labor/ladies_sampler.py#L25
import dgl.function as fn
import dgl
import torch
import numpy as np
import torch.nn.functional as F
import math

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

# # not used
# def normalized_edata(g, weight=None):
#     with g.local_scope():
#         if weight is None:
#             weight = 'W'
#             g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
#         g.update_all(fn.copy_e(weight, weight), fn.sum(weight, 'v'))
#         g.apply_edges(lambda edges: {'w': 1 / edges.dst['v']})
#         return g.edata['w']

class BanditSampler(dgl.dataloading.BlockSampler):
    def __init__(self, nodes_per_layer, importance_sampling=True, weight='w', out_weight='edge_weights', replace=False, eta=0.1):
        super().__init__()
        self.nodes_per_layer = nodes_per_layer
        self.importance_sampling = importance_sampling
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace
        self.eta = eta
    
    def compute_prob(self, g, seed_nodes, weight, num):
        """
        g : the whole graph
        seed_nodes : the output nodes for the current layer
        weight : the weight of the edges
        return : the unnormalized probability of the candidate nodes, as well as the subgraph
                 containing all the edges from the candidate nodes to the output nodes.
        """
        # create new subgraph using the incoming edges of the given nodes
        insg = dgl.in_subgraph(g, seed_nodes)
        # eliminate the isolated nodes across graph
        insg = dgl.compact_graphs(insg, seed_nodes)
        if self.importance_sampling:
            # reverse the edges (top-down)
            out_frontier = dgl.reverse(insg, copy_edata=True)
            # get the weights of the subgraph edges
            weight = weight[out_frontier.edata[dgl.EID].long()]
            # prob for each node wil be the sum of square edge weights for each node
            prob = dgl.ops.copy_e_sum(out_frontier, weight ** 2)
            # take the square root to follow the importance sampling equation
            prob = torch.sqrt(prob)
        else:
            # prob for choosing any neighbor is 1
            prob = torch.ones(insg.num_nodes())
            # set the non neighboring nodes to 0
            prob[insg.out_degrees() == 0] = 0
        return prob, insg

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
        # sample K nodes given the node probabilities,
        # where K is the passed num or the number of nodes in prob if num is larger than available nodes
        neighbor_nodes_idx = torch.multinomial(prob, min(num, prob.shape[0]), replacement=self.replace)
        return neighbor_nodes_idx
    
    def update_probability(prob, chosen_nodes, rewards, eta, k, n, T):
        """
        Update the probability of each node being selected using the rewards obtained
        from the previous selection using exp3 algo.

        Parameters
        ----------
        prob : numpy.ndarray
            Unnormalized probability distribution of nodes in the current subgraph.
        chosen_nodes : list
            The index of the nodes that was selected in the previous iteration.
        rewards : numpy.ndarray
            The rewards obtained for each node in the previous iteration.
        eta : float
            Learning rate for updating the probability.
        k : int
            Number of nodes to select in each iteration.
        n : int
            Number of nodes in the current subgraph.
        T : int
            Total number of iterations.

        Returns
        -------
        numpy.ndarray
            Updated unnormalized probability distribution.
        """
        # Update the probability distribution for each node
        # [?] the same as n, can we remove n?
        num_chosen = len(chosen_nodes)
        
        # total_reward = rewards.sum()
        # avg_reward = total_reward / num_chosen

        # Calculate the delta value for exp3
        delta = math.sqrt((1-eta)*eta**4*k**5*math.log(n/k)/(T*n**4))
        # delta = 1

        # Compute exponential weight for each chosen node
        # [!] add `rewards_hat = rewards / prob`, default `prob = 1/N_i` , where N_i is all neighbors of i
        # [?] should the equation be: `exp_weights = wij * torch.exp(delta * rewards_hat / n)`,
        # [!] add new parameter `weights`, default `wij = 1`.
        exp_weights = torch.exp(delta * rewards)

        # Update probability for chosen nodes and normalize
        for i in range(num_chosen):
            node = chosen_nodes[i]
            # [?] should the equation be: `prob[node] = (1 - eta) * (exp_weights[i] / sum(exp_weights)) + (eta / n)`
            prob[node] = (1 - eta) * (eta * (exp_weights[i] / sum(exp_weights))) + eta / n

        # # Normalize probability distribution
        # prob /= prob.sum()
        return prob

    # # not used
    # def select_node(self, g, prob):
    #     """
    #     Select a node from the graph based on the given probability distribution.
    #     Args:
    #         g (dgl.DGLGraph): The entire graph.
    #         prob (np.array): Probability distribution over the nodes.
    #     Returns:
    #         Tuple[int, int]: ID and index of the selected node.
    #     """
    #     print(prob)
    #     prob /= prob.sum()
    #     print(prob)
    #     nodes = np.arange(g.number_of_nodes())
    #     print(nodes)
    #     # return np.random.choice(nodes, p=prob)
    #     return torch.multinomial(prob, min(1, prob.shape[0]), replacement=self.replace)
    
    def generate_block(self, insg, neighbor_nodes_idx, seed_nodes, P_sg, W_sg):
        """
        insg : the subgraph yielded by compute_prob()
        neighbor_nodes_idx : the sampled nodes from the subgraph @insg, yielded by select_neighbors()
        seed_nodes_local_idx : the indices of seed nodes in the selected neighbor nodes, also yielded
                               by select_neighbors()
        P_sg : unnormalized probability of each node being sampled, yielded by compute_prob()
        W_sg : edge weights of @insg
        return : the block.
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
        # get the original node ids from insg
        nids = insg.ndata[dgl.NID][sg.ndata[dgl.NID].long()]

        # Normalize probability distribution
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

        # Convert the graph into a block
        block = dgl.to_block(sg, seed_nodes_idx.type(sg.idtype))
        # update the edge data with W_tilde
        block.edata[self.output_weight] = W_tilde

        # correct node ID mapping for source nodes
        block.srcdata[dgl.NID] = nids[block.srcdata[dgl.NID].long()]
        # correct node ID mapping for destination nodes
        block.dstdata[dgl.NID] = nids[block.dstdata[dgl.NID].long()]
        
        # get the original edge ids 
        sg_eids = insg.edata[dgl.EID][sg.edata[dgl.EID].long()]
        # set the original edge ids to the block
        block.edata[dgl.EID] = sg_eids[block.edata[dgl.EID].long()]
        return block

    def calculate_reward(self, insg, q):
        """
        Calculate the reward for each node in the graph @g, given the probability distribution @q.
        insg : the graph to compute rewards for
        q : the probability distribution over the nodes
        return : the reward for each node in the graph
        """
        # k is number of nodes in the subgraph
        k = insg.num_nodes()
        # get the device name
        device = q.device

        # calculate alpha_ij from Edge weights values of the batch graph, this weights can be different for each layer L
        alpha = torch.Tensor(insg.edata['w']).to(device) 

        # calculate ||h_j(t)|| 
        # h_j should be embedding of each node at layer L
        h_j = insg.ndata['nfeat'].to(device)
        # Compute the L2 norm of node embeddings
        h_j_norm = torch.norm(h_j, dim=1, keepdim=True) 
        # print(h_j_norm.shape)

        # calculate the reward
        # Because the edge weights are normalized, we don't need to divide by q_j
        # r = (alpha ** 2) / (k * (q_j ** 2)) * h_j_norm ** 2
        # r = dgl.ops.e_div_u(insg, (alpha ** 2), (k * (q ** 2)))
        # print(r)
        # print("R1: ", r.shape)
        # Compute reward as element-wise product of edge weight and node L2 norm squared
        r = dgl.ops.e_mul_v(insg, alpha ** 2, h_j_norm ** 2) 
        # print(r.shape)

        return r
    
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # copy seed_nodes to output_nodes (seed_nodes will be updated, output_nodes not)
        output_nodes = seed_nodes
        # convert seed_nodes IDs to tensor
        seed_nodes = torch.tensor(seed_nodes)
        # empty list 
        blocks = []
        # loop on the reverse of block IDs
        for block_id in reversed(range(len(self.nodes_per_layer))):
            # get the number of sample from nodes_per_layer per each block
            num_nodes_to_sample = self.nodes_per_layer[block_id]
            # get the edge weight from the original graph
            W = g.edata[self.edge_weight]
            # run compute_prob to get the unnormalized prob and subgraph
            prob, insg = self.compute_prob(g, seed_nodes, W, num_nodes_to_sample)
            # get cand_nodes IDs (all sampled nodes in the subgraph)
            cand_nodes = insg.ndata[dgl.NID]
            # print candidate nodes IDs
            print("candidate nodes: ", cand_nodes)
            
            # print the subgraph
            print("Insg: ", insg)
            # print seed_nodes IDs
            print("Seed nodes: ", seed_nodes)
            # print normalized prob
            print((prob/prob.sum()))

            # Apply bandit algorithm to choose the nodes
            # rewards = self.compute_rewards(insg, seed_nodes)
            rewards = self.calculate_reward(insg, prob) # needs to be in separate class
            print("Rewards: ", rewards)

            # chosen_node = self.select_node(insg, prob)
            # sample the best n neighbor nodes from given the probabilities of neighbors (and the current nodes)
            chosen_nodes = self.select_neighbors(prob, num_nodes_to_sample)
            
            # copy chosen_nodes to neighbor_nodes_idx
            # [?] can we remove this line?
            neighbor_nodes_idx = chosen_nodes
            # print chosen_nodes
            print("Choose node: ", chosen_nodes)
            # update nodes probabilities using EXP3 algorithm (given rewards)
            prob = self.update_probability(prob, chosen_nodes, rewards, self.eta, num_nodes_to_sample, insg.num_nodes(), T=100)
            # print updated prob after using EXP3
            print("Updated prob: ", prob)
            # neighbor_nodes_idx = insg.successors(chosen_node).t()
            
            # generate block for the sampled nodes and the previous nodes
            block = self.generate_block(insg, neighbor_nodes_idx.type(g.idtype), seed_nodes.type(g.idtype),
                                        prob, W[insg.edata[dgl.EID].long()])
            # update the seed_nodes with the sampled neighbors nodes to sample another block foe them in the next iteration
            seed_nodes = block.srcdata[dgl.NID]
            # add blocks at the beginning of blocks list (top-down)
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks
