
import os
import json
from data import GraphDataset
from network import get_network

num_trials = 5
num_folders = 25
folder_path = '../gnn8/last1/'
data_folder = '../data_folder/'
device = 'cpu'

dataset = GraphDataset(device=device)
dataset.load(data_folder+'/data/')
dataset2 = GraphDataset(device=device)
dataset2.load(data_folder+'/test/')


for i in range(num_folders):
    path = folder_path+f'output{i+1}/'
    files_names = os.listdir(path)
    models_path = [filee for filee in files_names if  filee.startswith("last_model_weights_trail")]
    args_path = path + [filee for filee in files_names if filee.startswith("Data_dataset_Hidden_")][0]

    with open(args_path, 'r') as f:
        args = json.load(f)
    args = args['hyper-parameters']

    model_op = get_network(args['architecture'])


def test():