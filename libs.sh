#!/bin/bash 

pip install scipy
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric

# for unsupervised graph transformer
apt-get update
apt-get install make
cd ..
cd GUAA/Downloads/Graph-Transformer-master/U2GNN_pytorch/log_uniform
python setup.py install