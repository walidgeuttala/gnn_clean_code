# Imports required libraries
import os
import torch

# Sets torch as the environment library
os.environ['TORCH'] = torch.__version__

# Imports the math library for mathematical functions
import math

# Fixes random seed
import random

# Imports libraries required for training and using the DGL library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(42)

# Imports DGL libraries for generating and training the model on Torch
import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import SAGEConv

# Imports libraries for calculations in the model
import numpy as np
import pandas as pd

# Imports networkx for generating graphs
import networkx as nx

# Imports libraries for handling plots
import matplotlib.pyplot as plt
import pylab

# Imports the statistics library for calculating mean
from statistics import mean

# Using binary search method
import bisect


import gzip

from data import GraphDataset
from create_dataset import *
from utils import *

import os
import json
import re
import shutil
import numpy as np
import torch
import numpy as np
from sklearn.decomposition import PCA


device = 'cpu'
dataset = GraphDataset(device=device)
dataset.load('./test')
from network import *
import os
import json
from main import parse_args

hidden_outputs = []

def hook(module, input, output):
    hidden_outputs.append(output)

@torch.no_grad()
def test2(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    argss = parse_args()
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)

        out, _ = model(batch_graphs, argss)
        pred = out.argmax(dim=1)
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()

        del out
        del pred

    return correct / num_graphs, loss / num_graphs


def count_output_folders():
    current_directory = os.getcwd()
    all_items = os.listdir(current_directory)
    output_folders = [folder for folder in all_items if os.path.isdir(folder) and folder.startswith('output')]
    return len(output_folders)



number_folders = count_output_folders()+1
number_folders = 2
current_path = '../density/'

results = []
selected_keys = ["architecture", "feat_type", "hidden_dim", "num_layers", "test_loss", "test_loss_error", "test_acc", "test_acc_error"]
for i in range(1, number_folders):
    #read the model
    output_path = current_path+"output{}/".format(i)
    # print(output_path)
    files_names = os.listdir(output_path)
    models_path = [filee for filee in files_names if  filee.startswith("last_model_weights_trail")]
    args_file_name = [filee for filee in files_names if filee.startswith("Data_dataset_Hidden_")][0]
    args_path = output_path+args_file_name

    with open(args_path, 'r') as f:
        args = json.load(f)
    args = args['hyper-parameters']

    accuracies = []
    losses = []
    model_op = get_network(args['architecture'])
    if args['feat_type'] == 'ones_feat':
        dataset.add_ones_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=100, shuffle=True)
    elif args['feat_type'] == 'degree_feat':
        dataset.add_degree_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=100, shuffle=True)
    elif args['feat_type'] == 'noise_feat':
        dataset.add_noise_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=100, shuffle=True)
    elif args['feat_type'] == 'identity_feat':
        dataset.add_identity_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=100, shuffle=True)
    else:
        dataset.add_normlized_degree_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=100, shuffle=True)
            
    num_feature, num_classes, _ = dataset.statistics()
    
    for num_trial, model_path in enumerate([models_path[0]]):
        print('{} {} {} {} {} {}'.format(num_feature, args['hidden_dim'], num_classes, args['num_layers'], args['dropout'], args['output_activation']))
        model = model_op(
                in_dim=num_feature,
                hidden_dim=args['hidden_dim'],
                out_dim=num_classes,
                num_layers=args['num_layers'],
                dropout=args['dropout'],
                output_activation = args['output_activation']
        ).to(device)
        model.load_state_dict(torch.load(output_path+model_path))
        model.eval()
        for name, layer in model.named_children():
            print(name)
        break
        accuracy, loss = test2(model, test_loader, device)
        accuracies.append(accuracy)
        losses.append(loss)

    accuracies = np.array(accuracies)
    losses = np.array(losses)

    result = [args["architecture"], args["feat_type"], args["hidden_dim"], args["num_layers"], losses.mean(), losses.var(), accuracies.mean(), accuracies.var()]
    result = dict(zip(selected_keys, result))
    results.append(result)

test_results = pd.DataFrame(results)
print(test_results)
test_results.to_csv('test_resutls_large_networks.csv')
