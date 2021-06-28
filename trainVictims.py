import argparse
import glob
import json
import os
import pickle
import random
import subprocess
import time
import warnings

import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as geo
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from models import Model

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#, 1, 2"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDBB')

def train(datasetName='IMDBB'):
    try:
        assert datasetName in ['IMDBB', 'DD', 'COLLAB', 'PROTEINS']
        print(datasetName)
    except AssertionError:
        raise NotImplementedError("Only IMDB-Binary, DD, COLLAB and PROTEINS dataset supported!")


    if datasetName == 'PROTEINS':
        runCommand = 'python HGP-SL-train.py'
    elif datasetName == 'IMDBB':
        runCommand = ''

    subprocess.run(runCommand, capture_output=True)
    # return trained model?

if __name__ == '__main__':
    args = parser.parse_args()
    datasetName = args.dataset.upper()
    train(datasetName)