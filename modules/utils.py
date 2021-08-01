import os
import pickle

import networkx
import torch
import torch_geometric as geo
from GraphTransformerPyTorch.pytorch_U2GNN_UnSup import *
from torch.nn.modules.transformer import Transformer


def getDataset(root, name, transform):
    if name.lower() in ['cora', 'pubmed', 'citeseer']:
        dataset = geo.datasets.Planetoid(
            root=root, name=name, transform=transform)
    elif name.lower() in ['mutag', 'imdb-binary', 'ethanol', 'proteins', 'dd']:
        dataset = geo.datasets.TUDataset(
            root=root, name=name, transform=transform, use_node_attr=True)
    else:
        raise NotImplementedError("{} not supported!".format(name))
    return dataset


def idx2adj(data):
    device = data.edge_index.device
    edge_index = torch.zeros(
        size=(data.num_nodes, data.num_nodes), device=device)
    for i in range(data.edge_index.shape[1]):
        edge_index[data.edge_index[0][i]][data.edge_index[1][i]] = 1
    return edge_index


def adj2idx(edge_index):
    assert edge_index.shape[0] == edge_index.shape[1]
    tmp = []
    for i in range(edge_index.shape[0]):
        for j in range(edge_index.shape[0]):
            if edge_index[i][j] == 1:
                tmp.append([i, j])
    return torch.Tensor(tmp).permute(1, 0).to(edge_index.device)


def batch_from_list(dataList):
    # generate a batched data from a list of geometric data
    # using this since wrong number of batch for surrogate data
    outp = geo.data.Batch()
    data = dataList[0]
    keys = data.keys
    keys.append('batch')
    batchedData = {key: [] for key in keys}
    cnt = 0
    prev_nodes = 0
    for data in dataList:
        for key in data.keys:
            if key == 'edge_index':
                data[key] += prev_nodes
            batchedData[key].append(data[key])
        batchedData['batch'].append(torch.ones(
            (data.x.shape[0], ), dtype=torch.int64) * cnt)
        prev_nodes += data.x.shape[0]
        cnt += 1
    for key in keys:
        if key == 'edge_index':
            catDim = -1
        else:
            catDim = 0
        outp[key] = torch.cat(batchedData[key], dim=catDim)
    return outp


def append(ori_graph, appendix, anchor_pos):
    # function to append a group of nodes to original graph
    idx_anchor = (ori_graph.edge_index[0] > anchor_pos).int().argmin() + 1
    first_half = ori_graph.edge_index[:, :idx_anchor]
    second_half = ori_graph.edge_index[:, idx_anchor + 1:]
    new_edge0 = torch.as_tensor([[anchor_pos], [ori_graph.num_nodes]],
                                device=ori_graph.x.device).long()
    new_edge1 = torch.as_tensor([[ori_graph.num_nodes], [anchor_pos]],
                                device=ori_graph.x.device).long()
    if appendix.num_nodes > 1:
        new_edge_index = torch.cat([first_half, new_edge0,
                                    second_half, new_edge1,
                                    appendix.edge_index + ori_graph.num_nodes],
                                   dim=1)
    else:
        new_edge_index = torch.cat(
            [first_half, new_edge0, second_half, new_edge1], dim=1)
    new_x = torch.cat([ori_graph.x, appendix.x], dim=0)
    ori_graph.edge_index = new_edge_index
    ori_graph.x = new_x
    ori_graph.num_nodes += appendix.num_nodes
    if ori_graph.edge_attr is not None:
        #         print("Appending to a weighted graph")
        first_half = ori_graph.edge_attr[:idx_anchor]
        second_half = ori_graph.edge_attr[idx_anchor + 1:]
        new_edge0 = torch.as_tensor([1.0], device=ori_graph.x.device)
        new_edge1 = torch.as_tensor([1.0], device=ori_graph.x.device)
        if appendix.num_nodes > 1:
            edge_attr = torch.cat(
                [first_half, new_edge0, second_half, new_edge1, appendix.edge_attr])
        else:
            edge_attr = torch.cat(
                [first_half, new_edge0, second_half, new_edge1])
        ori_graph.edge_attr = edge_attr
    return ori_graph


def BatchAppend(ori_graphs, trigger, pos_mode='deg'):
    # taking list of graphs, graph as input, return batch of graphes
    modi_graphs = []
    for i in range(len(ori_graphs)):
        tmp = ori_graphs[i].clone()
        pos_mode = pos_mode.lower()
        if pos_mode == 'deg':
            # append to node with the highest degree
            glueLoc = geo.utils.degree(tmp.edge_index[0]).argmax()
        elif pos_mode == 'min':
            # for experiment only, every other centrality measures take the highest value
            # append to node with the lowest degree
            glueLoc = geo.utils.degree(tmp.edge_index[0]).argmin()
        elif pos_mode == 'eig':
            # eigen vector centrality measures
            G = geo.utils.to_networkx(tmp)
            eigen_centrality = networkx.algorithms.centrality.eigenvector_centrality_numpy(
                G)
            glueLoc = sorted(eigen_centrality,
                             key=eigen_centrality.get, reverse=True)[0]
        elif pos_mode == 'btw':
            # betweenness centrality measures
            G = geo.utils.to_networkx(tmp)
            btw_centrality = networkx.algorithms.centrality.betweenness_centrality(
                G)
            glueLoc = sorted(
                btw_centrality, key=btw_centrality.get, reverse=True)[0]
        else:
            raise NotImplementedError(
                "Only degree, eigen-vector, and betweenness centrality measures!")
        modi_graph = append(tmp, trigger, glueLoc)
        modi_graphs.append(modi_graph)
#         print(modi_graph, modi_graphs)
#     modi_graphs = geo.data.Batch.from_data_list(modi_graphs).to(args.device)
    modi_graphs = batch_from_list(modi_graphs)
    return modi_graphs


def loadModel(datasetName):
    # unsupervised graph transformer models includes Cython modules, unpickle-able
    # Load model class for unsupervised graph transformer models first then load state dict
    if datasetName == 'PROTEINS':
        modelDir = os.path.join('/GUAA', 'models', datasetName, 'model.pth')
        model = torch.load(modelDir)
    else:
        with open(os.path.join(
                '/GUAA', 'models', datasetName, 'modelArguments.pickle'), 'rb') as handle:
            modelArguments = pickle.load(handle)
        model = TransformerU2GNN(**modelArguments)
        model.load_state_dict(torch.load(os.path.join(
            '/GUAA', 'models', datasetName, 'model.pth')))
    return model


def getNodes(numNodes, datasetName):
    if datasetName == 'PROTEINS':
        # node features are 4D vectors, the first dimension means the van de wall force and the next 3 are one-hot-encoded category
        tmp = torch.cat((torch.randint(-500, 800, (numNodes, 1)),
                         torch.nn.functional.one_hot(torch.randint(0, 3, (numNodes,)), num_classes=3)),
                        dim=1)
    else:
        # DD dataset, one-hot tensor of 89 labels
        tmp = torch.nn.functional.one_hot(
            torch.randint(0, 89, (numNodes, )), num_classes=89)
    return tmp.float().clone()

# function to transfer a torch geometric dataset to graph transformer readable data
# TODO


def geo2S2V(graphs, datasetName):
    '''
    desired format: 
    numGraphs
    numNodes, graphLabel
    str(nodeLabel, numNeighbors, [neibor index])
    details in https://github.com/daiquocnguyen/Graph-Transformer/issues/1
    '''
    # if datasetName in ['IMDBBINARY', 'COLLAB']:
    #     nodeLabel = 0
    # elif datasetName == 'DD':
    #     nodeLabelDict = list(range(83)) \
    # elif datasetName == 'PROTEINS':
    #     nodeLabelDict = list(range(3))
    outString = []
    outString.append(str(len(graphs))+'\n')
    for graph in graphs:
        nxg = geo.utils.convert.to_networkx(graph)
        outString.append(str(nxg.number_of_nodes())
                         + str(graph.y.item()) + '\n')
        for i in range(nxg.number_of_nodes()):
            try:
                nodeLabel = graph.x[i].argmax().item()
            except TypeError:
                # no node feature in this graph
                nodeLabel = 0
            neighborIndex = list(nxg.neighbors(i))
            numNeighbors = len(neighborIndex)
            neighborIndex.insert(0, nodeLabel)
            neighborIndex.insert(1, numNeighbors)
            neighborIndex = [str(idx) for idx in neighborIndex]
            outString.append(' '.join(neighborIndex)+'\n')
    print(outString)
    pass
