import argparse
import json
import logging
import os
from time import time
import numpy as np
import dgl
import numpy as np
import random
import torch
import torch.nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from network import get_network
from torch.utils.data import random_split
from utils import get_stats, boxplot, acc_loss_plot, set_random_seed
from data import GraphDataset
import h5py

def count_labels(indices, labels):
    label_counts = {}

    # Get unique labels
    unique_labels = np.unique(labels)

    # Initialize label counts
    for label in unique_labels:
        label_counts[label] = 0

    # Count labels for the given indices
    for index in np.nditer(indices):
        label = int(labels[index])
        label_counts[label] += 1

    return label_counts

def split_data_analysis():
    dataset = GraphDataset(device='cpu')
    dataset.load('./data')
    ans = []
    for seed in range(5):
        set_random_seed(seed)
        num_training = int(len(dataset) * 0.9)
        num_val = int(len(dataset) * 0.)
        num_test = len(dataset) - num_val - num_training
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(dataset, [num_training, num_val, num_test], generator=generator)
        ans.append(count_labels(np.array(train_set.indices), dataset.labels))
        ans.append(count_labels(np.array(test_set.indices), dataset.labels))
    ans = [list(d.values()) for d in ans]
    ans = np.array(ans)
    ans = (ans / 250) * 100
    return ans


print(split_data_analysis())
