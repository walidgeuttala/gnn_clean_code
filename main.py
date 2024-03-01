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


def parse_args():
    parser = argparse.ArgumentParser(description="GNN for network classification", allow_abbrev=False)
    parser.add_argument("--dataset", type=str, default="dataset", help="Just naming of the data added to the info after training the model")
    parser.add_argument("--dataset_path", type=str, default="../data_folder/data", help="Path to dataset")
    parser.add_argument("--test_dataset_path", type=str, default="../data_folder/test", help="Path to test dataset")
    parser.add_argument("--output_path", type=str, default="./output", help="Output path")
    # parser.add_argument("--plot_statistics", type=bool, default=False, help="Do plots about acc/loss/boxplot")
    parser.add_argument("--verbose", type=bool, default=True, help="print details of the training True or False")
    parser.add_argument("--device", type=str, default="cuda", help="Device cuda or cpu")
    parser.add_argument("--architecture",type=str,default="hierarchical",choices=["hierarchical", "global", "gat", "gin", "gatv2"],help="model architecture",)
    parser.add_argument("--data_type", type=str, default="regression", help="regression or classifcation")
    parser.add_argument("--label_type", type=str, default="original", choices=["original", "transitivity", "average_path", "density", "kurtosis"], help="choose")
    parser.add_argument("--feat_type", type=str, default="ones_feat", choices=["ones_feat", "noise_feat", "degree_feat", "identity_feat", "norm_degree_feat"], help="ones_feat/noies_feat/degree_feat/identity_feat")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay of the learning rate over epochs for the optimizer")
    parser.add_argument("--pool_ratio", type=float, default=0.2, help="Pooling ratio")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden size, number of neuron in every hidden layer but could change for currten type of networks")
    parser.add_argument("--dropout", type=float, default=0., help="Dropout ratio")
    parser.add_argument("--epochs", type=int, default=100, help="Max number of training epochs")
    parser.add_argument("--patience", type=int, default=-1, help="Patience for early stopping, -1 for no stop")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of conv layers")
    parser.add_argument("--print_every", type=int, default=1, help="Print train log every k epochs, -1 for silent training")
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials")
    parser.add_argument("--k", type=int, default=4, help="For ID-GNN where control the depth of the generated ID features for helping detecting cycles of length k-1 or less")
    parser.add_argument("--multi_k", type=bool, default=False, help="multiple feature type for non identity feature True or False")
    parser.add_argument("--output_activation", type=str, default="LogSoftmax", help="Output activation function")
    parser.add_argument("--optimizer_name", type=str, default="Adam", help="Optimizer type default adam")
    parser.add_argument("--loss_name", type=str, default='NLLLoss', help="Choose loss function correlated to the optimization function")
    parser.add_argument("--current_epoch", type=int, default=1, help="The current epoch")
    parser.add_argument("--current_trial", type=int, default=1, help="The current trial")
    parser.add_argument("--activate", type=bool, default=False, help="Activate the saving the node feature learned in the test dataset")
    parser.add_argument("--current_batch", type=int, default=1, help="The current batch")
    parser.add_argument("--changer", type=int, default=0, help="The current batch")
    # parser.add_argument("--save_hidden_output_train", type=bool, default=False, help="Saving the output before output_activation applied for the model in training")
    # parser.add_argument("--save_hidden_output_test", type=bool, default=False, help="Saving the output before output_activation applied for the model testing/validation")
    # parser.add_argument("--save_last_epoch_hidden_output", type=bool, default=False, help="Saving the last epoch hidden output only if it is false that means save for all epochs this applied to train and test if they are True")
    # parser.add_argument("--save_last_epoch_hidden_features_for_nodes", type=bool, default=False, help="Saving the last epoch hidden features of nodes only if it is false that means save for all epochs this applied to train and test if they are True")
    # parser.add_argument("--save_last_epoch_hidden_features_for_nodes", type=bool, default=False, help="Saving the last epoch hidden features of nodes only if it is false that means save for all epochs this applied to train and test if they are True")
    

    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    # print every
    if args.print_every == -1:
        args.print_every = args.epochs + 1

    # paths
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.patience == -1:
        args.patience = args.epochs+1

    name = "Data_{}_Hidden_{}_Arch_{}_Pool_{}_WeightDecay_{}_Lr_{}.log".format(
        args.dataset,
        args.hidden_dim,
        args.architecture,
        args.pool_ratio,
        args.weight_decay,
        args.lr,
        )    
    if args.changer == 1:
        name = "second_" + name
        
    args.output = os.path.join(args.output_path, name)
    
    if args.feat_type != 'identity_feat' and args.multi_k == False:
        args.k = 1

    return args


def train(model: torch.nn.Module, optimizer, trainloader, args):
    model.train()
    total_loss = 0.0
    num_graphs = 0
    num_batches = len(trainloader)
    loss_func = getattr(F, args.loss_name)(reduction="sum")
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(args.device)
        batch_labels = batch_labels.to(args.device)
    
        out = model(batch_graphs, args)
        loss = loss_func(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_graphs

@torch.no_grad()
def test_regression(model: torch.nn.Module, loader, args):
    model.eval()
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, args.loss_name)(reduction="sum")
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(args.device)
        batch_labels = batch_labels.to(args.device)
        out = model(batch_graphs, args)
        loss += loss_func(out, batch_labels).item()
        args.current_batch += 1

    return loss / num_graphs

@torch.no_grad()
def test_classification(model: torch.nn.Module, loader, args):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, args.loss_name)(reduction="sum")
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(args.device)
        batch_labels = batch_labels.to(args.device)
       
        out = model(batch_graphs, args)
        pred = out.argmax(dim=1)
        loss += loss_func(out, batch_labels).item()
        correct += pred.eq(batch_labels).sum().item()
        args.current_batch += 1
   
    return correct / num_graphs, loss / num_graphs

def main(args, seed, save=True):
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    set_random_seed(seed)
    dataset = GraphDataset(device=args.device)
    dataset2 = GraphDataset(device=args.device)
    dataset.load(args.dataset_path, args)
    dataset2.load(args.test_dataset_path, args)

    getattr(dataset, f'add_{args.feat_type}')(args.k)
    getattr(dataset2, f'add_{args.feat_type}')(args.k)

    test_loader2 = GraphDataLoader(dataset2, batch_size=args.batch_size, shuffle=False)
    num_training = int(len(dataset) * 0.9)
    num_val = int(len(dataset) * 0.)
    num_test = len(dataset) - num_val - num_training
    generator = torch.Generator().manual_seed(seed)
    train_set, _, test_set = random_split(dataset, [num_training, num_val, num_test], generator=generator)

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    
    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()
    args.num_feature = int(num_feature)
    args.num_classes = int(num_classes)
    set_random_seed(seed)
    model_op = get_network(args.architecture)
    model = model_op(
        in_dim=args.num_feature,
        hidden_dim=args.hidden_dim,
        out_dim=args.num_classes,
        num_layers=args.num_layers,
        pool_ratio=args.pool_ratio,
        dropout=args.dropout,
        output_activation = args.output_activation
    ).to(args.device)
    
    # Step 3: Create training components ===================================================== #
    if hasattr(torch.optim, args.optimizer_name):
        optimizer = getattr(torch.optim, args.optimizer_name)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # Replace `parameters` with your specific parameters
    else:
        print(f"Optimizer '{args.optimizer_name}' not found in torch.optim.")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Step 4: training epoches =============================================================== #
    train_times = []
    for e in range(args.epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader, args)
        scheduler.step()
        train_times.append(time() - s_time)

        if (e + 1) % args.print_every == 0 and args.verbose == True:
            log_format = ("Epoch {}: loss={:.4f}")
            print(log_format.format(e + 1, train_loss))
    if args.data_type == 'regression':
        test_acc = test_regression(model, test_loader, args)
        test_acc2 = test_regression(model, test_loader2, args)
    else:
        test_acc, _ = test_classification(model, test_loader, args)
        test_acc2, _ = test_classification(model, test_loader2, args)
    if save == True:
        if args.changer == 1:
            torch.save(model.state_dict(), '{}/last_second_model_weights_trail{}_{}_{}.pth'.format(args.output_path, seed, args.dataset, args.feat_type))
        else:
            torch.save(model.state_dict(), '{}/last_model_weights_trail{}_{}_{}.pth'.format(args.output_path, seed, args.dataset, args.feat_type))
    else:
        torch.save(model.state_dict(), '{}grid_search/{}_{}_{}_{}.pth'.format(args.output_path, args.architecture, args.feat_type, args.num_layers, args.hidden_dim))
   
    return test_acc, test_acc2, sum(train_times) / len(train_times)

if __name__ == "__main__":
    args = parse_args()
    accs = []
    accs2 = []
    train_times = []
    idx = 0
    best_acc = -1
    stat_list = []
    # train loss, train acc, valid acc, test acc
    list_results = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, acc2, train_time, = main(args, i)
        accs.append(acc)
        accs2.append(acc2)
        if best_acc < acc:
            idx = i
            best_acc = acc

        train_times.append(train_time)
    
    print("best trail model is : model_weights_trail{}_{}_{}.pth".format(idx, args.dataset, args.feat_type))

    mean, err_bd = get_stats(accs)
    print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))

    out_dict = {
        "hyper-parameters": vars(args),
        "result": "{:.4f}(+-{:.4f})".format(mean, err_bd),
        "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
        "results": [accs, accs2]
    }

    with open(args.output, "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)