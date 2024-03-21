import networkx as nx
import dgl

# Create a networkx graph
nx_graph = nx.Graph()
nx_graph.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

# Convert the networkx graph to a DGL graph
dgl_graph = dgl.from_networkx(nx_graph)

# Print the adjacency matrix of the DGL graph
print("Adjacency matrix of the DGL graph:")
print(dgl_graph.adjacency_matrix().to_dense())
