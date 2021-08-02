
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Dropout, Linear, MaxPool1d
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_sort_pool
from torch_geometric.utils import remove_self_loops

from modules.layers import GCN, HGPSLPool


class HGPSL(torch.nn.Module):
    def __init__(self, args):
        super(HGPSL, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio,
                               self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio,
                               self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(
            x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(
            x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)
        if not self.conv1.weight.requires_grad:
            return x.clone()
        else:
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = F.relu(self.lin2(x))
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = F.log_softmax(self.lin3(x), dim=-1)
            return x


class GCN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.conv5 = Conv1d(1, 16, 97, 97)
        self.conv6 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(352, 128)
        self.drop_out = Dropout(0.5)
        self.classifier_2 = Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        # TODO: add support for weighted graph
        x, edge_index, batch = data.x, data.edge_index, data.batch
        try:
            edge_weight = data.edge_attr
        except AttributeError:
            # no edge weight
            edge_weight = None
        edge_index, _ = remove_self_loops(edge_index)

        x_1 = torch.tanh(self.conv1(x, edge_index, edge_weight))
        x_2 = torch.tanh(self.conv2(x_1, edge_index, edge_weight))
        x_3 = torch.tanh(self.conv3(x_2, edge_index, edge_weight))
        x_4 = torch.tanh(self.conv4(x_3, edge_index, edge_weight))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = global_sort_pool(x, batch, k=30)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        classes = F.log_softmax(self.classifier_2(out), dim=-1)

        return classes
