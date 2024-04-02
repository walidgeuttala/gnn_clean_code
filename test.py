import dgl
import torch
import torch.nn as nn
from dgl.nn import GINEConv



g = dgl.graph(([0, 1, 2], [1, 1, 3]))
in_feats = 10
out_feats = 20
nfeat = torch.randn(g.num_nodes(), in_feats)
efeat = torch.randn(g.num_edges(), in_feats)
conv = GINEConv(nn.Linear(in_feats, out_feats))
res = conv(g, nfeat, efeat)
print(res.shape)