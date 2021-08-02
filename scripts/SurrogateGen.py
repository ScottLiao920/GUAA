import argparse
import glob
import json
import os
import pickle
import random
import time
import warnings

import GraphTransformerPyTorch
import modules.utils as utils
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geo
from torch._C import device
from torch.utils.data import random_split

import trainVictims

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # , 1, 2"


parser = argparse.ArgumentParser()
parser.add_argument('--datasetName', type=str, default='PROTEINS')
parser.add_argument('--surWeighted', type=bool, default=False)
parser.add_argument('--surDataperClass', type=int, default=500)
parser.add_argument('--maxStep', type=int, default=5000)
parser.add_argument('--trained', type=bool, default=True)
parser.add_argument('--device', type=str, default="CUDA:0")
args = parser.parse_args()

if torch.cuda.is_available() and args.device == 'CUDA:0':
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print("Performing calculations on {}".format(args.device))

if args.datasetName not in ['PROTEINS', 'IMDB-BINARY', 'DD', 'COLLAB']:
    raise NotImplementedError(
        "Only PROTEINS, IMDB-Binary, DD and COLLAB dataset supported!")
numNodes = 600

if not args.trained:
    print("\n\n\ntraining victim model first...\n\n\n")
    trainVictims.train(args.datasetName)

# TODO: make it returns a torch model (Done )
victimModel = utils.loadModel(args.datasetName).to(args.device)
victimModel.eval()
for param in victimModel.parameters():
    param.requires_grad = False

if args.datasetName == 'COLLAB':
    num_classes = 3
else:
    num_classes = 2

# surrogate data generation
# for every class, generate certain amount of class imporessions
# TODO: add weighted graph generation algorithms for PROTEINS & DD?
for cur_class in range(num_classes):
    # update class impressions individually
    for idx in range(args.surDataperClass):

        # create an random adjacency matrix for given nodes & labels
        num_nodes = random.randint(1, numNodes)
        sample = geo.data.Batch()
        sample.num_nodes = num_nodes

        # TODO: modify for every dataset
        if args.datasetName in ['PROTEINS', 'DD'] and not args.surWeighted:
            # not using node degree as node feature
            adv_adj = torch.zeros(
                size=(num_nodes, num_nodes),
                device=args.device).bool()  # create empty adjancency matrix
            for i in range(num_nodes):
                adv_adj[i, i:].random_(0, 2)  # randomly fill it
            adv_adj = adv_adj.int()
            if adv_adj.sum().item() == 0:
                idx -= 1
                continue  # if accidentally created an graph without edges, redo
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    adv_adj[j, i] = adv_adj[i, j]  # make it symmetric
            sample.edge_index = utils.adj2idx(adv_adj).long()
            sample.x = utils.getNodes(
                num_nodes, args.datasetName).to(args.device)
            sample.x.requires_grad_()
            cl_optim = torch.optim.Adam([sample.x], lr=0.1)
        elif args.datasetName in ['COLLAB', 'IMDB-BINARY']:
            # node degree as node feature
            adv_adj = torch.ones(size=(num_nodes, num_nodes)).int()
            sample.edge_index = utils.adj2idx(adv_adj).long()
            sample.edge_attr = torch.rand(sample.edge_index.shape[1], )
            sample.edge_attr.requires_grad_()
            print(sample.edge_attr.requires_grad)
            # diminish edges with weights lower than 0.5
            bin_attr = (sample.edge_attr + 0.5).int().bool()
            bin_edge0 = sample.edge_index[0].masked_select(bin_attr)
            bin_edge1 = sample.edge_index[1].masked_select(bin_attr)
            sample.x = torch.unsqueeze(bin_edge0.bincount().float(), 1)
            cl_optim = torch.optim.Adam([sample.edge_attr], lr=0.1)
        else:
            # pseudo-weighted graph generation for DD & PROTEINS
            adv_adj = torch.ones(size=(num_nodes, num_nodes),
                                 device=args.device).bool()
            sample.edge_index = geo.utils.remove_self_loops(
                utils.adj2idx(adv_adj).long())[0]
            sample.edge_attr = torch.rand(
                sample.edge_index.shape[1], ).to(args.device)
            sample.x = utils.getNodes(num_nodes, args.datasetName)
            sample.edge_attr.requires_grad_()
            cl_optim = torch.optim.Adam([sample.edge_attr], lr=0.1)

        cur_pred = victimModel(sample.to(args.device))
        cur_tar = random.uniform(0.55, 0.99)
        cnt = 0
        while F.softmax(cur_pred)[:, cur_class].item() < cur_tar and cnt < args.maxStep:
            cl_optim.zero_grad()
            cur_pred = victimModel(sample.to(args.device))
            loss = F.nll_loss(cur_pred, torch.Tensor(
                [cur_class]).long().to(args.device))
            loss.backward()
            cl_optim.step()
            if cnt % 500 == 0:
                if args.datasetName in ['PROTEINS', 'DD'] and not args.surWeighted:
                    print(sample.x.grad.sum().item())
                else:
                    print(sample.edge_attr.grad.sum().item())
                print('{} |　Epoch {} | Target Class {} | Current Logits for target class {}'.format(
                    idx, cnt, cur_class, F.softmax(cur_pred)[0, cur_class].item()))
            cnt += 1
            with torch.no_grad():
                if args.datasetName == 'PROTEINS':
                    sample.x.clamp_(-500, 600)
                elif args.datasetName == 'DD':
                    # for DD dataset, performs an inplace softmax operation to get it within 0 to 1
                    torch.exp(sample.x, out=sample.x)
                    summed = torch.sum(sample.x, dim=1, keepdim=True)
                    sample.x /= summed
        # TODO: modify generated surrogate data to one-hot or keep them in softmax-ed manner?
        torch.save(sample, os.path.join(
            'data', args.datasetName, 'classImpression', str(cur_class), '{}.pt'.format(idx))
        )
        print('Epoch {} | Target Class {} | Current Logits for target class {}'.format(
            cnt, cur_class, F.softmax(cur_pred)[0, cur_class].item()))
