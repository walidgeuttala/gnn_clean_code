import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import os
import random
from sklearn.metrics import roc_auc_score

import dgl.data

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def compute_loss(pos_g_data, neg_g_data, features, device, get_auc=True):
    pos_u, pos_v = pos_g_data.edges()
    pos_nodes_first = torch.index_select(features.cpu(), 0, pos_u.to(features.device).cpu())
    pos_nodes_second = torch.index_select(features.cpu(), 0, pos_v.to(features.device).cpu())

    pos_pred = torch.sum(pos_nodes_first.cpu() * pos_nodes_second.cpu(), dim=-1)

    neg_u, neg_v = neg_g_data.edges()
    neg_nodes_first = torch.index_select(features.cpu(), 0,neg_u.to(features.device).cpu())
    neg_nodes_second = torch.index_select(features.cpu(), 0,neg_v.to(features.device).cpu())
    #
    neg_pred = torch.sum(neg_nodes_first.cpu() * neg_nodes_second.cpu(), dim=-1)

    labels = torch.cat([torch.ones(pos_pred.shape[0]), torch.zeros(neg_pred.shape[0])]).long().to(device)

    loss_func = nn.BCEWithLogitsLoss()
    loss = loss_func(torch.cat([pos_pred, neg_pred]).cpu(), labels.float().cpu())

    if get_auc:
        auc = compute_auc(pos_pred, neg_pred)
        return loss, auc
    else:
        return loss

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

from dgl.nn.pytorch import SAGEConv,GraphConv

# 定义一个两层的GraphSage模型
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 定义一个两层的GraphSage模型
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, weight=True, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, h_feats, weight=True, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


import dgl.function as fn

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # 通过源节点特征“h”和目标节点特征“h”之间的点积计算两点之间存在边的Score
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v为每条边返回一个元素向量，因此需要squeeze操作
            return g.edata['score'][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph."""


for repeat in range(5):
    print('Repeat {}: '.format(repeat))
    # setup_seed(args.seed, torch.cuda.is_available())
    setup_seed(42)

    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]

    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # 采样所有负样例并划分为训练集和测试集中。
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    model = GCN(1433, 16)
    model = model.to(device)
    pred = DotPredictor()
    # compute_loss = nn.BCEWithLogitsLoss()
    # compute_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

    all_logits = []

    train_g = train_g.to(device)
    train_pos_g = train_pos_g.to(device)
    train_neg_g = train_neg_g.to(device)
    test_pos_g = test_pos_g.to(device)
    test_neg_g = test_neg_g.to(device)

    for run in range(1000):
        # 前向传播
        h = model(train_g, train_g.ndata['feat'])
        # pos_score = pred(train_pos_g, h)
        # neg_score = pred(train_neg_g, h)

        # loss = compute_loss(pos_score, neg_score)
        loss, auc_train = compute_loss(train_pos_g, train_neg_g, h, device)

        # if run % 5 == 0:
        #     print('In epoch {}, loss: {}'.format(run, loss))

        if run % 100 == 0:
            # print('In epoch {}, loss: {}'.format(run, loss))
            with torch.no_grad():
                # pos_score = pred(test_pos_g, h)
                # neg_score = pred(test_neg_g, h)
                # auc = compute_auc(pos_score, neg_score)
                loss2, auc = compute_loss(test_pos_g, test_neg_g, h, device)

                print('Run {}, test_acc: {:.16f}'.format(run, auc))

        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
