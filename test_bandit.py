import dgl
import numpy as np

def select_node(g, prob):
    """
    Select a node from the graph based on the given probability distribution.
    Args:
        g (dgl.DGLGraph): The entire graph.
        prob (np.array): Probability distribution over the nodes.
    Returns:
        Tuple[int, int]: ID and index of the selected node.
    """
    prob = prob / np.sum(prob)
    nodes = np.arange(g.number_of_nodes())
    return np.random.choice(nodes, p=prob)

def update_probability(prob, chosen_node, rewards, eta, node_ids):
    """
    Update the probability distribution using the EXP3 algorithm.
    Args:
        prob (np.array): Probability distribution over the nodes.
        chosen_node (int): ID of the node that was selected.
        rewards (np.array): Reward obtained for each node in the previous round.
        eta (float): Learning rate.
        node_ids (np.array): Array containing the IDs of the nodes in the subgraph.
    Returns:
        np.array: Updated probability distribution over the nodes.
    """
    num_nodes = len(prob)
    weight = np.exp(eta * rewards / num_nodes)
    prob[chosen_node] = (1 - eta) * prob[chosen_node] + eta * (weight / np.sum(weight))
    prob /= np.sum(prob)
    new_prob = np.zeros(num_nodes)
    new_prob[node_ids] = prob
    return new_prob

def generate_mini_batch(g, num_samples):
    """
    Generate mini-batch nodes for a layer in GCN.
    Args:
        g (dgl.DGLGraph): The entire graph.
        num_samples (int): Number of nodes to sample for mini-batch.
    Returns:
        (dgl.DGLGraph, np.array): The subgraph containing the mini-batch nodes and
                                   their IDs in the original graph.
    """
    nodes = np.arange(g.number_of_nodes())
    np.random.shuffle(nodes)
    sample_nodes = nodes[:num_samples]
    sample_nodes.sort()
    sg = g.subgraph(sample_nodes)
    return sg, sample_nodes

# Generate a sample graph
num_nodes = 10
from scipy.sparse import coo_matrix

# create a random adjacency matrix
adj_matrix = np.random.randint(2, size=(num_nodes, num_nodes))
adj_coo = coo_matrix(adj_matrix)

# create a DGL graph from the adjacency matrix
g = dgl.from_scipy(adj_coo)
print(g)

# Parameters for the EXP3 algorithm
num_samples = 6
eta = 0.1
num_rounds = 2

# Initialize the probability distribution
prob = np.ones(g.number_of_nodes()) / g.number_of_nodes()
for t in range(num_rounds):
    sg, node_ids = generate_mini_batch(g, num_samples)
    print(node_ids)
    degs = sg.in_degrees().float().numpy()
    norm = np.sum(np.power(degs, -0.5))
    norm = np.max([norm, 1])
    prob_sg = np.ones(num_samples) / num_samples
    chosen_nodes = []
    i = 0
    while i < num_samples:
        n_id = node_ids[i]
        if np.isin(n_id, g.nodes()):
            prob_sg[i] = prob[n_id]
        selected_node = select_node(sg, prob_sg)
        if selected_node not in chosen_nodes:
            chosen_nodes.append(selected_node)
            i += 1
    print(chosen_nodes)