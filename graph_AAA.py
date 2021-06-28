import os
import torch
import torch.nn as nn


import torchvision
import torchvision.transforms as transforms
import scipy.io as sio

import torch_geometric as geo

import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()