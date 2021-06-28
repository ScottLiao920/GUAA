import argparse
import glob
import json
import os
import pickle
import random
import time
import warnings

import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geo
from torch.utils.data import random_split

from models import Model
import utils
import trainVictims

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#, 1, 2"


parser = argparse.ArgumentParser()
parser.add_argument('--datasetName', type=str, default='IMDBB')
parser.add_argument('--surDataperClass', type=int, default=500)
parser.add_argument('--maxStep', type=int, default=5000)
parser.add_argument('--trained', type=bool, default=False)
args = parser.parse_args()

# TODO: modify the average number of nodes in each dataset
if args.datasetName == 'PROTEINS':
    numNodes = 600
elif args.datasetName == 'IMDBB':
    numNodes = 40
elif args.datasetName == 'DD':
    numNodes = 80
elif args.datasetName == 'COLLAB':
    numNodes = 100
else:
    raise NotImplementedError("Only PROTEINS, IMDB-Binary, DD and COLLAB dataset supported!")

if not args.trained:
    print("\n\n\ntraining victim model first...\n\n\n")
    trainVictims.train(args.datasetName)

victimModel = utils.loadModel(args.datasetName) # TODO
victimModel.eval()
for param in victimModel.parameters():
    param.requires_grad = False
    print(param.requires_grad)

dataset = utils.getDataset('data', args.datasetName, None)

# surrogate data generation
# for every class, generate certain amount of class imporessions
for cur_class in range(dataset.num_classes):
    # update class impressions individually
    for idx in range(args.surDataperClass):
        
        # create an random adjacency matrix for given nodes & labels
        num_nodes = random.randint(1, numNodes)
        sample = geo.data.Batch()

        if args.datasetName not in ['COLLAB', 'IMDBB']:
            # not using node degree as node feature
            adv_adj = torch.zeros(size=(num_nodes, num_nodes), device=args.device).bool()
            for i in range(num_nodes):
                adv_adj[i, i:].random_(0, 2)
            adv_adj = adv_adj.int()
            if adv_adj.sum().item() == 0: 
                idx -= 1
                continue
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    adv_adj[j, i] = adv_adj[i, j]
            sample.edge_index = utils.adj2idx(adv_adj).long()
            sample.x = utils.getNodes(num_nodes)
            sample.x.requires_grad_()
            cl_optim = torch.optim.Adam([sample.x], lr=0.1)
        else:
            adv_adj = torch.ones(size=(num_nodes, num_nodes)).int()
            sample.edge_index = utils.adj2idx(adv_adj).long()
            sample.num_nodes = num_nodes
            sample.edge_attr = torch.rand(sample.edge_index.shape[1], )
            sample.edge_attr.requires_grad_()
            bin_attr = (sample.edge_attr + 0.5).int().bool()
            bin_edge0 = sample.edge_index[0].masked_select(bin_attr)
            bin_edge1 = sample.edge_index[1].masked_select(bin_attr)
            sample.x = bin_edge0.bincount().float()
            cl_optim = torch.optim.Adam([sample.edge_attr], lr=0.1)

        cur_pred = victimModel(sample)
        cur_tar = random.uniform(0.55, 0.99)
        cnt = 0
        while F.softmax(cur_pred)[:, cur_class].item() < cur_tar and cnt < args.maxStep: 
            cl_optim.zero_grad()
            cur_pred = victimModel(sample)
            loss = F.nll_loss(cur_pred, torch.Tensor([cur_class]).long().to(args.device))
            loss.backward()
            cl_optim.step()
            if cnt % 500 == 0:
                print(sample.x.grad.sum())
                print('{} |　Epoch {} | Target Class {} | Current Logits for target class {}'.format(
                    idx, cnt, cur_class, F.softmax(cur_pred)[0,cur_class].item()))
            cnt += 1
            with torch.no_grad():
                # TODO: modify the lower and upper limits for each dataset
                sample.x.clamp_(-500, 600)
        torch.save(sample, os.path.join(
            'data', args.datasetName, 'classImpression', str(cur_class), '{}.pt'.format(idx))
        )
        print('Epoch {} | Target Class {} | Current Logits for target class {}'.format(
                    cnt, cur_class, F.softmax(cur_pred)[0,cur_class].item()))
