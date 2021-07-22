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

from modules.models import Model

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # , 1, 2"
os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='IMDBB')


def train(datasetName='IMDBB'):
    try:
        assert datasetName in ['IMDBB', 'DD', 'COLLAB', 'PROTEINS']
        print(datasetName)
    except AssertionError:
        raise NotImplementedError(
            "Only IMDB-Binary, DD, COLLAB and PROTEINS dataset supported!")

    if datasetName == 'PROTEINS':
        runCommand = '; '.join([
            'cd modules'
            'python HGP-SL-train.py'
        ])
    elif datasetName == 'IMDBB':
        tmpCommand = ['cd Graph-Transformer-PyTorch',
                      '''
        python train_pytorch_U2GNN_UnSup.py \
            --dataset IMDBBINARY \
            --batch_size 4 \
            --ff_hidden_size 1024 \
            --fold_idx 1 \
            --num_neighbors 8 \
            --num_epochs 50 \
            --num_timesteps 4 \
            --learning_rate 0.0005 \
            --model_name IMDBBINARY_bs4_fold1_1024_8_idx0_4_1
        '''
                      ]
        runCommand = '; '.join(tmpCommand)
    elif datasetName == 'DD':
        tmpCommand = ['cd Graph-Transformer-PyTorch',
                      '''
        python train_pytorch_U2GNN_UnSup.py \
            --dataset DD \
            --batch_size 4 \
            --ff_hidden_size 64 \
            --fold_idx 1 \
            --num_neighbors 4 \
            --num_epochs 50 \
            --num_timesteps 4 \
            --learning_rate 0.0005 \
            --model_name DD_bs4_fold1_64_4_idx0_4_1
        '''
                      ]
        runCommand = '; '.join(tmpCommand)
    elif datasetName == 'COLLAB':
        tmpCommand = ['cd Graph-Transformer-PyTorch',
                      '''
        python train_pytorch_U2GNN_UnSup.py \
            --dataset COLLAB \
            --batch_size 4 \
            --ff_hidden_size 1024 \
            --fold_idx 1 \
            --num_neighbors 8 \
            --num_epochs 50 \
            --num_timesteps 4 \
            --learning_rate 0.0005 \
            --model_name COLLAB_bs4_fold1_1024_8_idx0_4_1
        '''
                      ]
        runCommand = '; '.join(tmpCommand)
    else:
        raise NotImplementedError
    p = subprocess.Popen(runCommand, shell=True)
    p.communicate()
    print("Finished training!")
    # return trained model?


if __name__ == '__main__':
    args = parser.parse_args()
    datasetName = args.dataset.upper()
    train(datasetName)
