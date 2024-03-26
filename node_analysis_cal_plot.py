import json

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

# Load arguments from JSON file
def load_args_from_json(args_dict):
    return Args(**args_dict)


from network_node_analysis import *

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
from dgl.dataloading import GraphDataLoader

from data import *
from identity import *
# Imports DGL libraries for generating and training the model on Torch
import dgl
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

from utils import *
from data import *
import os
import json
import re
import shutil
import numpy as np
import torch
import numpy as np
from sklearn.decomposition import PCA
import dgl
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from main import parse_args



number_folders = 21
#print(number_folders)
current_path = '../gnn_outputs/version1/kurtosis/'
out = 0
device = 'cpu'



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
    out = []
    for batch in loader:
        batch_graphs, batch_labels = batch
        plot_degree_distribution(batch_graphs)
        g = batch_graphs.cpu().to_networkx()
        g = nx.Graph(g)
        degrees = np.array(list(dict(g.degree()).values()))
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)

        if i in [3, 7, 11, 15, 19]:
          outt, graph = model(batch_graphs, args)
          g = graph.cpu().to_networkx()
          g = nx.Graph(g)
          degrees = np.array(list(dict(g.degree()).values()))
        else:
          outt = model(batch_graphs, argss)
        out.append(outt)
        break

    return out, degrees

def plot_degree_distribution(graph):
    # Calculate the degree of each node in the graph
    degrees = graph.in_degrees().numpy()

    # Plot the degree distribution
    plt.hist(degrees, bins=np.arange(max(degrees) + 2) - 0.5, color='blue', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(True)
    plt.show()

def count_output_folders():
    current_directory = os.getcwd()
    all_items = os.listdir(current_directory)
    output_folders = [folder for folder in all_items if os.path.isdir(folder) and folder.startswith('output')]
    return len(output_folders)


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

    # Example usage
    args2 = load_args_from_json(args)

    dataset = GraphDataset(device=device)
    dataset.load('../data_folder/test', args2)

    accuracies = []
    losses = []
    model_op = get_network(args['architecture'])
    if args['feat_type'] == 'ones_feat':
        dataset.add_ones_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    elif args['feat_type'] == 'degree_feat':
        dataset.add_degree_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    elif args['feat_type'] == 'noise_feat':
        dataset.add_noise_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    elif args['feat_type'] == 'identity_feat':
        dataset.add_identity_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataset.add_norm_degree_feat(args['k'])
        test_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    degrees = 0
    for num_trial, model_path in enumerate([models_path[0]]):
        model = model_op(
                in_dim=args['num_feature'],
                hidden_dim=args['hidden_dim'],
                out_dim=args['num_classes'],
                num_layers=args['num_layers'],
                dropout=args['dropout'],
                output_activation = args['output_activation']
        ).to(device)
        model.load_state_dict(torch.load(output_path+model_path))
        model.eval()

        out, degrees= test2(model, test_loader, device)
        break
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming your 2D matrix is named 'data_matrix' and the degree list is named 'degree_list'
    # 'data_matrix' is a 2D numpy array where rows represent nodes and columns represent features
    # 'degree_list' is a list containing the degree of each node
    # Here's a sample 'data_matrix' and 'degree_list' for illustration purposes


    data_matrix = out[0].numpy()  # 100 nodes with 10 features each
    degree_list = degrees  # Example degree list for 100 nodes
    print(data_matrix.shape)
    # Calculate minimum, mean, and maximum values for each node
    min_values = np.min(data_matrix, axis=1)
    mean_values = np.mean(data_matrix, axis=1)
    max_values = np.max(data_matrix, axis=1)

    # Sort the node indices based on degree
    sorted_indices = np.argsort(degree_list)
    sorted_degree = degree_list[sorted_indices]
    sorted_min_values = min_values[sorted_indices]
    sorted_mean_values = mean_values[sorted_indices]
    sorted_max_values = max_values[sorted_indices]

    # Plot the results
    num_nodes = data_matrix.shape[0]
    node_indices = np.arange(1, num_nodes + 1)
    model_name = args['architecture']
    feat_name = args['feat_type']

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 2)
    #plt.scatter(node_indices, sorted_min_values, label='Minimum', marker='o')
    plt.scatter(node_indices, sorted_mean_values, label='Value', marker='x')
    #plt.scatter(node_indices, sorted_max_values, label='Maximum', marker='^')
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.title(f'Values with Node Degree {model_name} {feat_name} (Ordered by Degree)')
    plt.legend()
    plt.subplot(2, 1, 1)
    plt.scatter(node_indices, sorted_degree, label='Degree', marker='x', color='black')  # Scatter plot for degree
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.title(f'Values with Node Degree {model_name}  {feat_name} (Ordered by Degree)')
    plt.legend()
    #plt.yscale('log', base=2)
    plt.xticks(node_indices)
    
    plt.savefig(f'{current_path}/{model_name}_{feat_name}.png')
    plt.close()

