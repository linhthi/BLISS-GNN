# Example from https://docs.dgl.ai/en/2.0.x/stochastic_training/node_classification.html

# Install required packages.
import os
import torch
import numpy as np
os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

# Install the CPU version. If you want to install CUDA version, please
# refer to https://www.dgl.ai/pages/start.html.
device = torch.device("cuda:2")
# !pip install --pre dgl -f https://data.dgl.ai/wheels-test/repo.html

try:
    import dgl
    import dgl.graphbolt as gb
    installed = True
except ImportError as error:
    installed = False
    print(error)
print("DGL installed!" if installed else "DGL not found!")

dataset = gb.BuiltinDataset("ogbn-arxiv").load()

graph = dataset.graph
feature = dataset.feature
train_set = dataset.tasks[0].train_set
valid_set = dataset.tasks[0].validation_set
test_set = dataset.tasks[0].test_set
task_name = dataset.tasks[0].metadata["name"]
num_classes = dataset.tasks[0].metadata["num_classes"]
print(f"Task: {task_name}. Number of classes: {num_classes}")

datapipe = gb.ItemSampler(train_set, batch_size=1024, shuffle=True)
datapipe = datapipe.sample_neighbor(graph, [4, 4])
datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
datapipe = datapipe.copy_to(device)
train_dataloader = gb.DataLoader(datapipe, num_workers=0)

# data = next(iter(train_dataloader))
# print(data)


import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type="mean")
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h = self.conv1(mfgs[0], x)
        h = F.relu(h)
        h = self.conv2(mfgs[1], h)
        return h


in_size = feature.size("node", None, "feat")[0]
model = Model(in_size, 256, num_classes).to(device)

opt = torch.optim.Adam(model.parameters())

datapipe = gb.ItemSampler(valid_set, batch_size=1024, shuffle=False)
datapipe = datapipe.sample_neighbor(graph, [4, 4])
datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
datapipe = datapipe.copy_to(device)
valid_dataloader = gb.DataLoader(datapipe, num_workers=0)


import sklearn.metrics

import tqdm

for epoch in range(100):
    model.train()

    with tqdm.tqdm(train_dataloader) as tq:
        for step, data in enumerate(tq):
            x = data.node_features["feat"]
            labels = data.labels

            predictions = model(data.blocks, x)

            loss = F.cross_entropy(predictions, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            accuracy = sklearn.metrics.accuracy_score(
                labels.cpu().numpy(),
                predictions.argmax(1).detach().cpu().numpy(),
            )

            tq.set_postfix(
                {"loss": "%.03f" % loss.item(), "acc": "%.03f" % accuracy},
                refresh=False,
            )

    model.eval()

    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for data in tq:
            x = data.node_features["feat"]
            labels.append(data.labels.cpu().numpy())
            predictions.append(model(data.blocks, x).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print("Epoch {} Validation Accuracy {}".format(epoch, accuracy))
