import dgl
import torch
import dgl.function as fn

# Create a simple graph with 3 nodes and 2 edges
edges = [(0, 1), (1, 2)]
g = dgl.graph(edges)

# Add node features, all ones
h = torch.ones(3, 1)
g.ndata['h'] = h

# Add edge weights, with weight = 1
g.edata['weight'] = torch.ones(g.num_edges(), 1)

# Create a graph with self-loops using DGL's built-in function
g_with_self_loops = dgl.add_self_loop(g)

# Perform aggregation on the graph with self-loops
g_with_self_loops.ndata['h'] = h  # Reset features to original ones
g_with_self_loops.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
print("Aggregated node features with self-loops:\n", g_with_self_loops.ndata['h'])

# Manually adding self features without self-loops
g.ndata['h'] = h  # Reset features to original ones
g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
g.ndata['h'] += h  # Adding self features manually
print("Aggregated node features without self-loops but with manual self addition:\n", g.ndata['h'])
