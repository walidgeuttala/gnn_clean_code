import networkx as nx
import dgl
import torch
from data import GraphDataset
from network import *
from utils import *
from dgl.dataloading import GraphDataLoader

import argparse
import json
import logging
import os
from time import time

import numpy as np
import dgl
import h5py
import networkx as nx
import torch
import torch.nn as nn
import torch.nn as F
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

from network import get_network
from utils import get_stats, boxplot, acc_loss_plot, set_random_seed
from data import GraphDataset



@torch.no_grad()
def test_regression(model: torch.nn.Module, loader):
    model.eval()
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, 'MSELoss')(reduction="sum")
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += 1
        out = model(batch_graphs)
        loss += loss_func(out, batch_labels).item()

    return loss / num_graphs

def main(seed=1):
    set_random_seed(seed)
    graphs = []
    n = 8
    degree = 4
    device = 'cpu'
    # ER_low
    graphs.append(nx.gnp_random_graph(n, degree / n, seed=1, directed=False))
    print('adj matrix : ')
    print(nx.adjacency_matrix(graphs[0]).toarray())
    labels = torch.tensor([sum(dict(nx.degree(graphs[0])).values()) / len(graphs[0])]).unsqueeze(1)
    print('labels : ', labels)
    degrees = torch.ones(8, 1).float()
    print(degrees)
    #nx.set_node_attributes(graphs[0], degrees, 'feat')
    graphs = [dgl.from_networkx(graph) for graph in graphs]
    graphs[0].ndata['feat'] = degrees
    dataset = GraphDataset(graphs, labels, device)

    test_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    
    
    # Step 2: Create model =================================================================== #
    set_random_seed(seed)
    model = SAGNetworkGlobal(
            in_dim = 1,
            hidden_dim = 2,
            out_dim = 1,
            num_layers=1,
            pool_ratio = 0.0,
            dropout = 0.0,
            output_activation = 'Identity'
        )
    for name, param in model.named_parameters():
        print(name, param.data)
    # Step 3: Create training components ===================================================== #
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=0.2, weight_decay=0.0)  # Replace `parameters` with your specific parameters


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    
    
    loss_value = test_regression(model, test_loader)
        
    return loss_value

print(main())