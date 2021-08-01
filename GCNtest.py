import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Dropout, Linear, MaxPool1d
from torch.utils import data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import (GCNConv, GraphConv, global_mean_pool,
                                global_sort_pool)
from torch_geometric.utils import remove_self_loops

from modules.utils import Indegree
from modules.models import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--datasetName', type=str, default='IMDBBINARY')
args = parser.parse_args()


dataset = TUDataset(root='data/GCN', name=args.datasetName,
                    pre_transform=Indegree(), use_node_attr=True)
torch.manual_seed(0)
dataset = dataset.shuffle()

train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8) + 1:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = GCN(num_features=dataset.num_node_features,
            num_classes=dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss()


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        # Perform a single forward pass.
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # Check against ground-truth labels.
        correct += int((pred == data.y).sum())
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)


bestAccu = 0
esCnt = 0
for epoch in range(1, 501):
    train()
    with torch.no_grad():
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        if test_acc >= bestAccu:
            bestAccu = test_acc
            esCnt = 0
            outpDir = os.path.join(
                '/GUAA', 'models', args.datasetName, 'model.pt')
            try:
                torch.save(model.state_dict(), outpDir)
            except FileNotFoundError:
                # need to create sub-dir first
                os.mkdir(os.path.join(
                    '/GUAA', 'models', args.datasetName))
                torch.save(model.state_dict(), outpDir)
        else:
            esCnt += 1
        if esCnt >= 30:
            break
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
print("Best Accuracy: %.4f" % bestAccu)
