import torch as th
import dgl
from dgl.data import DGLDataset

def load_data(data):
    g = data[0]
    g.ndata['features'] = g.ndata.pop('feat').bfloat16()
    g.ndata['labels'] = g.ndata.pop('label')
    return g, data.num_classes

def load_dgl(name):
    from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset, RedditDataset, YelpDataset, FlickrDataset, ActorDataset

    d = {
        'cora': CoraGraphDataset,
        'citeseer': CiteseerGraphDataset,
        'pubmed': PubmedGraphDataset,
        'reddit': RedditDataset,
        'yelp': YelpDataset,
        'flickr': FlickrDataset,
        'actor': ActorDataset
    }

    return load_data(d[name]())

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=self_loop)
    return load_data(data)


def load_ogb(name, root="dataset"):
    from ogb.nodeproppred import DglNodePropPredDataset

    print("load", name)
    data = DglNodePropPredDataset(name=name, root=root)
    print("finish loading", name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata.pop('feat').bfloat16()
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    graph.ndata['labels'] = labels.type(th.LongTensor)
    in_feats = graph.ndata['features'].shape[1]

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask
    print("finish constructing", name)
    return graph, num_labels

def load_dataset(dataset_name):
    multilabel = False
    if dataset_name in ['reddit', 'cora', 'citeseer', 'pubmed', 'yelp', 'flickr', 'actor']:
        g, n_classes = load_dgl(dataset_name)
        multilabel = dataset_name in ['yelp']
        if multilabel:
            g.ndata['labels'] = g.ndata['labels'].to(dtype=th.float32)
    elif dataset_name in ['ogbn-products', 'ogbn-arxiv', 'ogbn-papers100M']:
        g, n_classes = load_ogb(dataset_name)
    elif dataset_name == 'toy':
        dataset = ToyDataset()
        g, n_classes, multilabel = dataset[0], 2, multilabel
    else:
        raise ValueError('unknown dataset')
    
    return g, n_classes, multilabel

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata["train_mask"])
    val_g = g.subgraph(g.ndata["train_mask"] | g.ndata["val_mask"])
    test_g = g
    return train_g, val_g, test_g


class ToyDataset(DGLDataset):
    def __init__(self):
        super().__init__(name="toy")

    def process(self):
        self.graph = dgl.graph(([2, 3, 3, 4], [0, 0, 1, 1] ), num_nodes=5, col_sorted=True)
        self.graph.ndata["features"] = th.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]] , dtype=th.float32)
        self.graph.ndata["labels"] = th.tensor([0, 0, 1, 1, 1] , dtype= th.float).type(th.LongTensor)
        self.graph.edata["weight"] = th.FloatTensor([0.5, 0.5, 0.3, 0.7])

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = 5
        n_train = 5
        n_val = 0
        train_mask = th.zeros(n_nodes, dtype=th.bool)
        val_mask = th.zeros(n_nodes, dtype=th.bool)
        test_mask = th.zeros(n_nodes, dtype=th.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1