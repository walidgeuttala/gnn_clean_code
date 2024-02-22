import numpy as np

import torch
import torch.nn as nn
import torch.nn as F

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, MaxPooling
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import GINConv, GraphConv
from dgl.nn import GATv2Conv

from layer import ConvPoolBlock, SAGPool

import h5py


class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The input node feature dimension.
        hidden_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_layers (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        output_activation = 'log_softmax'
    ):
        super(SAGNetworkHierarchical, self).__init__()

        self.dropout = dropout
        self.num_convpools = num_layers
        self.output_activation = output_activation
        convpools = []
        for i in range(num_layers):
            _i_dim = in_dim if i == 0 else hidden_dim
            _o_dim = hidden_dim
            convpools.append(
                ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio)
            )
        self.convpools = torch.nn.ModuleList(convpools)

        self.mlp = MLP(hidden_dim * 2, hidden_dim, out_dim)
        self.output_activation = getattr(nn, self.output_activation)(dim=-1)

    def forward(self, graph: dgl.DGLGraph, args):
        feat = graph.ndata["feat"]
        final_readout = None
        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat, args)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout

        feat = self.mlp(final_readout)

        return self.output_activation(feat)

# hidden_dim is the feat output
class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`

    Args:
        in_dim (int): The input node feature dimension.
        hidden_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_layers (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        output_activation = 'log_softmax'
    ):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_activation = output_activation
        convs = []
        for i in range(num_layers):
            _i_dim = in_dim if i == 0 else hidden_dim
            _o_dim = hidden_dim
            convs.append(GraphConv(_i_dim, _o_dim, allow_zero_in_degree=True))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_layers * hidden_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.mlp = MLP(concat_dim * 2, hidden_dim, out_dim)
        self.output_activation = getattr(nn, self.output_activation)(dim=-1)

    def forward(self, graph: dgl.DGLGraph, args):
        feat = graph.ndata["feat"]
        conv_res = []

        for i in range(self.num_layers):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)
     
        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat(
            [self.avg_readout(graph, feat), self.max_readout(graph, feat)],
            dim=-1,
        )

        feat = self.mlp(feat)

        return self.output_activation(feat)

#hideen_feat is the output dim
class GAT(torch.nn.Module):
    """
    A graph neural network (GAT) that performs graph sum pooling over all nodes in each layer and makes a prediction
    using a linear layer.

    Args:
        num_layers (int): Number of layers in the GAT
        hidden_dim (int): Hidden dimension of the GAT layers
        drop (float): Dropout probability to use during training (default: 0)

    Attributes:
        layers (nn.ModuleList): List of GAT layers
        num (int): Number of layers in the GAT
        input_dim (int): Dimension of the input feature vector
        output_dim (int): Dimension of the output prediction vector
        linear_prediction (nn.ModuleList): List of linear layers to make the prediction
        pool (SumPooling): A sum pooling module to perform graph sum pooling

    Methods:
        forward(g, h): Perform a forward pass through the GAT given a graph and input node features.

    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        output_activation = 'log_softmax',
    ):
        """
        Initializes a new instance of the GAT class.

        Args:
            num_layers (int): Number of layers in the GAT
            hidden_dim (int): Hidden dimension of the GAT layers
            drop (float): Dropout probability to use during training (default: 0)

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.output_activation = output_activation
        self.ann_input_shape = num_layers * hidden_dim
        self.num_heads = 4
        self.batch_norms = []
        # Create GAT layers
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                conv = GATv2Conv(in_feats=self.input_dim,out_feats=hidden_dim, num_heads=self.num_heads, activation=nn.ReLU(), allow_zero_in_degree=True, share_weights=True)
            else:
                conv = GATv2Conv(in_feats=hidden_dim*self.num_heads, out_feats=hidden_dim, num_heads=self.num_heads, activation=nn.ReLU(), allow_zero_in_degree=True,share_weights=True)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim*self.num_heads))
            self.layers.append(conv)

        # Create linear prediction layers
        self.linear_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            _i_dim = hidden_dim*self.num_heads
            _o_dim = hidden_dim
            self.linear_prediction.append(torch.nn.Sequential(torch.nn.Linear(_i_dim, _o_dim),
                                        torch.nn.ReLU(),
                                        torch.nn.BatchNorm1d(_o_dim)))


        self.mlp = MLP(hidden_dim*num_layers, hidden_dim, out_dim)
        # Create sum pooling module

        self.pool = SumPooling()
        self.output_activation = getattr(nn, self.output_activation)(dim=-1)

    def forward(self, graph: dgl.DGLGraph, args):
        """
        Perform a forward pass through the GAT given a graph and input node features.

        Args:
            g (dgl.DGLGraph): A DGL graph
            h (torch.Tensor): Input node features

        Returns:
            score_over_layer (torch.Tensor): Output prediction

        """
        # list of hidden representation at each layer
        feat = graph.ndata["feat"]
        # Compute hidden representations at each layer
        pooled_h_list = []
        for i, layer in enumerate(self.layers):
            feat = layer(graph, feat).flatten(1)
            self.batch_norms[i] = self.batch_norms[i].to(args.device)
            feat = self.batch_norms[i](feat)
            pooled_h = self.pool(graph, feat)
            pooled_h_list.append(self.linear_prediction[i](pooled_h))

        pooled_h = torch.cat(pooled_h_list, dim=-1)
        pooled_h = self.mlp(pooled_h)

        return self.output_activation(pooled_h)

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
        self.relu = nn.ReLU()
    def forward(self, x):
        h = x
        h = self.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

class GATv2(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers=3,
        pool_ratio: float = 0.5,
        dropout: float = 0.0,
        output_activation = 'log_softmax',
    ):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = torch.nn.ReLU()
        num_hidden = hidden_dim
        heads = [2] * num_layers
        feat_drop = 0
        attn_drop = 0
        negative_slope = 0.2
        residual = False
        num_classes = out_dim
        self.output_activation = output_activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                allow_zero_in_degree=True,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATv2Conv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    bias=False,
                    allow_zero_in_degree=True,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                allow_zero_in_degree=True,
                bias=False,
                share_weights=True,
            )
        )
        self.mlp = MLP(num_classes, num_classes, out_dim)
        # Create sum pooling module

        self.pool = SumPooling()
        self.output_activation = getattr(nn, self.output_activation)(dim=-1)

    def forward(self, g, args):
        h = g.ndata["feat"]
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gatv2_layers[-1](g, h).mean(1)
        logits = self.pool(g, logits)
        logits = self.mlp(logits)

        return self.output_activation(logits)


class GIN(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers = 5,
                 pool_ratio=0,
                 dropout=0.,
                 output_activation = 'log_softmax'):

        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.output_activation = output_activation

        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            #if layer == 0:
            #    print(mlp.linears[0].weight)
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers + 1):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(in_dim, hidden_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.mlp = MLP(hidden_dim, hidden_dim, out_dim)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
        self.relu = nn.ReLU()
        self.output_activation = getattr(nn, self.output_activation)(dim=-1)

    def forward(self, g, args):
        # list of hidden representation at each layer (including the input layer)
        h = g.ndata["feat"]
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = self.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0

        # perform graph sum pooling over all nodes in each layer
        pooled_h_list = []
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            pooled_h_list.append(pooled_h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        score_over_layer = self.mlp(score_over_layer)
        return  self.output_activation(score_over_layer)




def get_network(net_type: str = "hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    elif net_type == 'gat':
        return GAT
    elif net_type == 'gin':
        return GIN
    elif net_type == 'gatv2':
        return GATv2
    else:
        raise ValueError(
            "Network type {} is not supported.".format(net_type)
        )