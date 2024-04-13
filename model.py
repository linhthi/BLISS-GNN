import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch as th
import torch.nn as nn
import tqdm

import dgl
import dgl.nn as dglnn


# source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gatv2
class GATv2(nn.Module):
    def __init__(
            self, num_layers, in_dim, num_hidden, num_classes, heads, activation, feat_drop,
            attn_drop, negative_slope, residual, allow_zero_in_degree
            ):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.heads = heads

        allow_zero_in_degree = True

        if num_layers > 1:
            # input projection (no residual)
            self.gatv2_layers.append(dglnn.GATv2Conv(
                in_dim, num_hidden, heads[0], feat_drop, attn_drop, negative_slope, False,
                self.activation, bias=False, share_weights=True, allow_zero_in_degree=allow_zero_in_degree))
            # hidden layers
            for l in range(1, self.num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gatv2_layers.append(dglnn.GATv2Conv(
                    num_hidden * heads[l - 1], num_hidden, heads[l], feat_drop, attn_drop, negative_slope, residual,
                    self.activation, bias=False, share_weights=True, allow_zero_in_degree=allow_zero_in_degree))
            # output projection
            self.gatv2_layers.append(dglnn.GATv2Conv(
                num_hidden * heads[-2], num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual,
                None, bias=False, share_weights=True, allow_zero_in_degree=allow_zero_in_degree))
        else:
            self.gatv2_layers.append(dglnn.GATv2Conv(
                in_dim, num_classes, heads[-1], feat_drop, attn_drop, negative_slope, residual,
                None, bias=False, share_weights=True, allow_zero_in_degree=allow_zero_in_degree))


    def forward(self, blocks, inputs):
        h = inputs
        # print(-1, h.shape)
        for l, block in enumerate(blocks):
            # print(f'block_{l}.srcdata', block.srcdata[dgl.NID], block.srcdata[dgl.NID].shape)
            # print(f'block_{l}.edata', block.edata[dgl.EID], block.edata[dgl.EID].shape)

            # save the mag of (h) into block.srcdata
            block.srcdata['embed_norm'] = th.reshape(th.norm(h, dim=1, keepdim=True), (-1,))
            # print(f'h_norm_{l}', block.srcdata['embed_norm'], block.srcdata['embed_norm'].shape)
            h, a = self.gatv2_layers[l](block, h, get_attention=True)
            a = th.mean(a.squeeze(dim=-1), dim=1) # average attention weights across heads
            # print(f'a_{l}', a, a.shape)
            block.edata['a_ij'] = a
            if l < len(blocks) - 1:
                h = h.flatten(1)
            else:
                # logits
                h = h.mean(1)
        # output projection
        return h
    
    def inference(self, g, device, batch_size, use_uva, num_workers):
        """
        Inference with the GATv2 model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        g.ndata["h"] = g.ndata["features"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["h"]
        )
        pin_memory = g.device != device and use_uva
        dataloader = dgl.dataloading.DataLoader(
            g,
            th.arange(g.num_nodes(), dtype=g.idtype, device=g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            use_uva=use_uva,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        self.eval()

        for l, layer in enumerate(self.gatv2_layers):
            y = th.empty(
                g.num_nodes(),
                self.num_hidden * self.heads[l] if l < len(self.gatv2_layers) - 1 else self.num_classes,
                dtype=g.ndata["h"].dtype,
                device=g.device,
                pin_memory=pin_memory,
            )
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)
                if l < len(self.gatv2_layers) - 1:
                    h = h.flatten(1)
                else:
                    h = h.mean(1)
                # by design, our output nodes are contiguous
                y[output_nodes[0].item() : output_nodes[-1].item() + 1] = h.to(
                    y.device
                )
            g.ndata["h"] = y
        return y

class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # for i in block.srcdata:
            #     print('block.srcdata_' + str(i), block.srcdata[i],  block.srcdata[i].shape)
            # save the mag of (h) into block.srcdata
            block.srcdata['embed_norm'] = th.reshape(th.norm(h, dim=1, keepdim=True), (-1,))
            h = layer(block, h, edge_weight=block.edata['edge_weights'] if 'edge_weights' in block.edata else None)
            if l < len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    # TODO: check devices L290 h = layer(blocks[0], x)
    def inference(self, g, device, batch_size, use_uva, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        g.ndata["h"] = g.ndata["features"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["h"]
        )
        pin_memory = g.device != device and use_uva
        dataloader = dgl.dataloading.DataLoader(
            g,
            th.arange(g.num_nodes(), dtype=g.idtype, device=g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            use_uva=use_uva,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        self.eval()

        for l, layer in enumerate(self.layers):
            y = th.empty(
                g.num_nodes(),
                self.n_hidden if l < len(self.layers) - 1 else self.n_classes,
                dtype=g.ndata["h"].dtype,
                device=g.device,
                pin_memory=pin_memory,
            )
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)
                if l < len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0].item() : output_nodes[-1].item() + 1] = h.to(
                    y.device
                )
            g.ndata["h"] = y
        return y


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class="multinomial", max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average="micro")
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average="micro")
    return f1_micro_eval, f1_micro_test
