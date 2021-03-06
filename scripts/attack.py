import argparse
import glob
import json
import os
import pickle
import random
import time
import warnings

from networkx.generators.geometric import soft_random_geometric_graph

import modules.utils as utils
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geo
from torch._C import device, dtype
from torch.utils.data import random_split

import trainVictims
random.seed(0)
torch.manual_seed(0)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # , 1, 2"


parser = argparse.ArgumentParser()
parser.add_argument('--datasetName', type=str, default='IMDB-BINARY',
                    choices=['IMDB-BINARY', 'DD', 'COLLAB', 'PROTEINS'])
parser.add_argument('--triggerLen', type=int, default=3)
parser.add_argument('--dataRoot', type=str, default='data/GCN')
parser.add_argument('--posMode', type=str, default='all',
                    choices=['deg', 'min', 'eig', 'btw', 'all'])
parser.add_argument('--trainingData', dtype=str, default='all',
                    choices=['actual', 'sur', 'all'])
parser.add_argument('--embedMethod', dtype=str, default='all',
                    choices=['DGE', 'NF', 'all'])
parser.add_argument('--device', type=str, default="CUDA:0")
args = parser.parse_args()


def attack(advDataSet):
    result = {}
    for triggerLen in range(1, 4):
        for posMode in posModes:
            log = {}
            start = time.time()
            graph_idx = list(range(len(advDataSet)))
            trigger_idx = random.choices(range(len(triggerList)), k=triggerLen)
            while len(graph_idx) > 0:
                trigger = utils.getTrigger(
                    n=triggerLen, idx=trigger_idx, weighted=True)
                cur_batch = random.choices(graph_idx, k=16)
                graph_idx = [tmp for tmp in graph_idx if tmp not in cur_batch]
                modi_graph = utils.BatchAppend([advDataSet[i]
                                                for i in cur_batch], trigger)
                modi_graph.x.requires_grad_()
                advOutput, embedAdv = victimModel(modi_graph)

                ori_graph = geo.data.Batch.from_data_list(
                    [advDataSet[i] for i in cur_batch]).to(args.device)
                ori_graph.x.requires_grad_()
                labels = geo.data.Batch.from_data_list(
                    [advDataSet[i] for i in cur_batch]).y

                loss = F.nll_loss(output, labels)
                grad_embed_adv = torch.autograd.grad(
                    loss, modi_graph.x)  # , allow_unused=True)

                output, embed_ori = victimModel(ori_graph)
                loss = F.nll_loss(output, labels)
                grad_embed_ori = torch.autograd.grad(
                    loss, ori_graph.x)  # , allow_unused=True)

                for i in range(len(trigger_idx)):
                    cur_score = torch.ones((len(triggerList, )))
                    for triggerCand in range(len(triggerList)):
                        # node embedding via difference of graph embedding
                        cur_score[triggerCand] = torch.matmul((
                            triggerList[triggerCand].x -
                            modi_graph.x[-(triggerLen-i)]).mean(dim=0).unsqueeze(0),
                            (grad_embed_adv[0][-(triggerLen-i)]).unsqueeze(-1)).item()
                    trigger_idx[i] = cur_score.argmin(-1).item()
            time_used = time.time() - start
            trigger = utils.getTrigger(
                n=triggerLen, idx=trigger_idx, weighted=True)
            modi_graph = utils.BatchAppend(test_set, trigger, posMode)
            output, embed_adv = victimModel(modi_graph)
            accu = (output.argmax(-1) == geo.data.Batch.from_data_list(
                test_set).y).int().sum().item() / len(test_set)
            log['trigger_idx'] = trigger_idx
            log['accu'] = accu
            log['timeCost'] = time_used
            result['length: {} mode: {}'.format(triggerLen, posMode)] = log


transform = utils.Indegree() if args.datasetName in [
    'IMDB-BINARY', 'COLLAB'] else None
dataset = utils.getDataset(args.dataRoot, args.datasetName, transform)
num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(
    dataset, [num_training, num_val, num_test])

# now need graph embedding outcome for comparison
victimModel = utils.loadModel(args.datasetName, outpEmbed=True)
posModes = [['deg', 'min', 'eig', 'btw']
            if args.posMode == 'all' else [args.posMode]]


# get all posible candidates for hot-flip inspired selection
cands = {}
for i in range(len(dataset)):
    for j in range(dataset[i].x.shape[0]):
        key = '{} {}'.format(
            dataset[i].x[j, 0].item(), dataset[i].x[j, 1:].argmax().item())
        try:
            cands[key] += 1
        except KeyError:
            cands[key] = 1
# put them in a single list
triggerList = []
for i in range(len(cands.keys())):
    triggerList.append(utils.getTrigger(1, [i]))

if args.trainingData != 'actual':
    # load generated class impressions
    cldataList = []
    for i in range(dataset.num_classes):
        label = torch.Tensor([i]).long()
        clDir = os.path.join(args.dataRoot, args.dataset_name,
                             'classImpression', str(i)+'_tropology')
        for fin in os.listdir():
            tmp = torch.load(os.path.join(clDir, fin))
            tmp.y = label
            tmp = geo.data.Data(x=tmp.x, edge_index=tmp.edge_index,
                                edge_attr=tmp.edge_attr, y=tmp.y)
            cldataList.append(tmp.to(args.device))
            if (tmp.edge_index.max().item() - tmp.x.shape[0]) != -1:
                print(fin, i)
            if (tmp.edge_index.shape[1] != tmp.edge_attr.shape[0]):
                print(fin, i)
    attack(cldataList)
    if args.trainingData == 'all':
        attack(training_set)
else:
    attack(training_set)

