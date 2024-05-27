import os
from time import time

import dgl
import torch
import torch.nn as nn
import torch.nn as F
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

import random
import numpy as np
import os
import torch
import dgl 
from dgl.data import DGLDataset
from dgl import load_graphs
from dgl.data.utils import load_info
from identity import compute_identity
from dgl.nn import AvgPooling, GINConv
from torch.nn.functional import relu

class GraphFeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, graph_loader):
        # Concatenate all node features into a single tensor
        all_feats = torch.cat([graph.ndata['feat'] for graph, _ in graph_loader], dim=0)
        
        # Compute mean and standard deviation
        self.mean = torch.mean(all_feats, dim=0)
        self.std = torch.std(all_feats, dim=0)
        
        # Normalize all node features across all graphs
        for graph, _ in graph_loader:
            graph.ndata['feat'] = (graph.ndata['feat'] - self.mean) / self.std

    def transform(self, graph_loader):
        # Check if the mean and std have been computed
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std have not been computed. Please fit the normalizer first.")
        
        # Transform the node features of test data using the computed mean and std
        for graph, _ in graph_loader:
            graph.ndata['feat'] = (graph.ndata['feat'] - self.mean) / self.std

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)

def print_epsilon(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer.gin_conv, GINConv):
            print(f'Layer {i} epsilon: {layer.gin_conv.eps.item()}')

device = "cuda"
dataset_path  = "../data_folder/data"
k = 1
feat_type = "ones_feat"
batch_size = 100
optimizer_name = "Adam"
lr = 0.1
weight_decay = 0.95
epochs = 100
hidden_dim = 1
num_layers = 1
loss_name = "MSELoss"


# create a DGLDataset for our graphs and labels
class GraphDataset(DGLDataset):
    '''
    GraphDataset is a custom dataset class that inherits from DGLDataset.
    It is designed to store and process graph data for machine learning tasks.
    
    Parameters:
    graphs (list): a list of DGL graphs
    labels (torch.tensor): a tensor of labels
    
    '''
    def __init__(self, graphs=None, labels=None, device='cpu'):
        self.graphs = graphs
        self.labels = labels 
        self.device = device
        self.data_path = None
        self.properties = [ 'average_path_labels', 'transitivity_labels', 'kurtosis_labels', 'density_labels']
        self.data_types = ['classification', 'regression']
        
    def __len__(self):
        '''
        Returns:
        int: the length of the dataset
        '''
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''
        Returns the data at the specified index.
        
        Parameters:
        idx (int): the index to retrieve data from
        
        Returns:
        tuple: a tuple containing the graph and label at the specified index
        '''
        return self.graphs[idx], self.labels[idx]

    def load(self, data_path):
        '''
        Loads the processed data from disk as .bin and .pkl files. The processed data consists of the graph data and the corresponding labels.
        '''
        # Load the graph data and labels from the .bin file
        graph_path = os.path.join('{}/dgl_graph.bin'.format(data_path))
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        # Load the other information about the dataset from the .pkl file
        info_path = os.path.join('{}/info.pkl'.format(data_path))
        self.gclasses = load_info(info_path)['gclasses']
        self.dim_nfeats = load_info(info_path)['dim_nfeats']
        #self.device = load_info(info_path)['device']
        self.data_path = data_path
        #self.choose_labels(data_path+'/properties_labels.pt')
        
        if self.device == 'cuda':
            self.graphs = [g.to(self.device) for g in self.graphs]
            
        #self.add_self_loop()
        self.degree_label()
        if self.device == 'cuda': 
            self.labels = self.labels.to(self.device)

    def load2(self, data_name):
        '''
        Loads the processed data from disk as .bin and .pkl files. The processed data consists of the graph data and the corresponding labels.
        '''
        # Load the graph data and labels from the .bin file
        dataset = dgl.data.TUDataset(data_name)
        self.graphs = []
        for idx in range(len(dataset)):
            self.graphs.append(dataset[idx][0])
        #self.choose_labels(f"../data_folder/dgl_graph_labels/{data_name}_properties_labels.pt")
        if self.device == 'cuda':
            self.graphs = [g.to(self.device) for g in self.graphs]
        #self.add_self_loop()
        self.degree_label()
        if self.device == 'cuda':
            self.labels = self.labels.to(self.device)


    def degree_label(self):
        self.labels = []
        for idx in range(len(self.graphs)):
            self.labels.append(self.graphs[idx].num_edges() / self.graphs[idx].num_nodes() + 1)
        self.labels = torch.tensor(self.labels).view(-1, 1).float()

    def has_cache(self):
        '''
        Checks if the processed data has been saved to disk as .bin and .pkl files.
        '''
        # Check if the .bin and .pkl files for the processed data exist in the directory
        graph_path = os.path.join(f'{self.data_path}/dgl_graph.bin')
        info_path = os.path.join(f'{self.data_path}/info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def add_self_loop(self):
        for idx in range(len(self.graphs)):
            self.graphs[idx] = self.graphs[idx].add_self_loop()
    
    def choose_labels(self, file_path):
        # started from 1 as the first labels is the original label
        self.labels = torch.load(file_path)
        self.labels = self.labels[-1].view(-1, 1).float()
        self.labels += 1

    def add_ones_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            g.ndata['feat'] = torch.ones(g.num_nodes(), k).float().to(self.device)
    def add_noise_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs: 
            g.ndata['feat'] = torch.rand(g.num_nodes(), k).float().to(self.device)
            
    def add_degree_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            degrees = g.out_degrees().unsqueeze(1).float().to(self.device)
            repeated_degrees = degrees.repeat(1, k)  # Repeat degree 'k' times
            g.ndata['feat'] = repeated_degrees

    def add_identity_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            g.ndata['feat'] = compute_identity(torch.stack(g.edges(), dim=0), g.number_of_nodes(), k).float().to(self.device)

    def add_norm_degree_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            degrees = g.in_degrees().unsqueeze(1).float().to(self.device)
            repeated_degrees = degrees.repeat(1, k) / (g.number_of_nodes() - 1) # Repeat degree 'k' times
            g.ndata['feat'] = repeated_degrees

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv

from dgl.utils import expand_as_pair

class SimpleGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SimpleGNNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.eps = 0

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Use DGL's built-in sum aggregation
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            #h = g.ndata["h"]
            h = h +  g.ndata["h"]

            return self.linear(h)

class SimpleGNNLayer2(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats):
        super(SimpleGNNLayer2, self).__init__()
        self.linear1 = nn.Linear(in_feats, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_feats, bias=False)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Use DGL's built-in sum aggregation
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = g.ndata["h"]
            h = self.linear1(h)  # Apply first linear layer with ReLU activation
            h = self.linear2(h)  # Apply second linear layer
            return h

class SingleLayerGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SingleLayerGNN, self).__init__()
        self.layer = SimpleGNNLayer(in_dim, out_dim)
        self.pool = (AvgPooling()) 

    def forward(self, g):
        h = g.ndata['feat']
        h = self.layer(g, h)
        h = self.pool(g, h)
        return h

    def forward2(self, g):
        h = g.ndata['feat']
        h = self.layer(g, h)
        h_pooled = self.pool(g, h)
        return h_pooled, h

class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, output_dim, bias=False))
        #self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        #self.relu = nn.ReLU()
    def forward(self, x):
        h = x
        #h = self.relu(self.linears[0](h))
        return self.linears[0](h)

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim1 = in_dim 
            else:
                in_dim1 = hidden_dim
            if i+1 == num_layers:
                out_dim1 = out_dim
            else:
                out_dim1 = hidden_dim
            self.gnn_layers.append(GINConv(MLP(in_dim1, hidden_dim, out_dim1), init_eps=0, learn_eps=False))
            #self.gnn_layers.append(SimpleGNNLayer2(in_dim1, hidden_dim, out_dim1))
            #self.gnn_layers.append(SimpleGNNLayer(in_dim1, out_dim1))

        self.pool = (AvgPooling()) 

    def forward(self, g):
        h = g.ndata['feat']
        for i in range(num_layers):
            h = self.gnn_layers[i](g, h)
        h = self.pool(g, h)
        return h

    # def forward2(self, g):
    #     h = g.ndata['feat']
    #     for i in range(num_layers):
    #         h = self.gnn_layers[i](g, h)
    #     h_pooled = self.pool(g, h)
    #     return h_pooled, h
    

def train(model: torch.nn.Module, optimizer, trainloader):
    model.train()
    total_loss = 0.0
    num_graphs = 0
    
    loss_func = getattr(F, loss_name)(reduction="sum")
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        num_graphs += batch_size
    
        out = model(batch_graphs)
        loss = loss_func(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_graphs

@torch.no_grad()
def test_regression(model: torch.nn.Module, loader):
    model.eval()
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, loss_name)(reduction="sum")
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_size
        out = model(batch_graphs)
        loss += loss_func(out, batch_labels).item()

    return loss / num_graphs

@torch.no_grad()
def test_regression2(model: torch.nn.Module, loader):
    model.eval()
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, loss_name)(reduction="sum")
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += 1
        out, hidden = model.forward2(batch_graphs)
        print("value out", out)
        print("value hidden", hidden)
        loss += loss_func(out, batch_labels).item()
        break
    return loss / num_graphs

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def plot_degree_distribution(graph):
    # Calculate the degree of each node in the graph
    degrees = graph.cpu().out_degrees().numpy()

    # Plot the degree distribution
    plt.hist(degrees, bins=np.arange(max(degrees) + 2) - 0.5, color='blue', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(True)
    plt.savefig(f'degree_dist_samples.png')
    plt.show()

@torch.no_grad()
def test_regression_sample(model: torch.nn.Module, loader, samples):
    model.eval()
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, loss_name)(reduction="sum")
    degrees = 0
    out = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += samples
        out, outt = model.forward2(batch_graphs)
        loss += loss_func(out, batch_labels).item()
        out = outt
        plot_degree_distribution(batch_graphs)
        g = batch_graphs.cpu().to_networkx()
        g = nx.Graph(g)
        degrees = np.array(list(dict(g.degree()).values()))
        break

    # Assuming your 2D matrix is named 'data_matrix' and the degree list is named 'degree_list'
    # 'data_matrix' is a 2D numpy array where rows represent nodes and columns represent features
    # 'degree_list' is a list containing the degree of each node
    # Here's a sample 'data_matrix' and 'degree_list' for illustration purposes


    data_matrix = out.cpu().numpy()  # 100 nodes with 10 features each
    degree_list = degrees  # Example degree list for 100 nodes
    print(data_matrix.shape)
    print(degrees.shape)
    # Calculate minimum, mean, and maximum values for each node
    # min_values = np.min(data_matrix, axis=1)
    # mean_values = np.mean(data_matrix, axis=1)
    # max_values = np.max(data_matrix, axis=1)

    # # Sort the node indices based on degree
    # sorted_indices = np.argsort(degree_list)
    # sorted_degree = degree_list[sorted_indices]
    # sorted_min_values = min_values[sorted_indices]
    # sorted_mean_values = mean_values[sorted_indices]
    # sorted_max_values = max_values[sorted_indices]

    # Plot the results
    num_nodes = data_matrix.shape[0]
    node_indices = np.arange(1, num_nodes + 1)
    model_name = "GIN"
    feat_name = "degree_feat"

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 2)
    #plt.scatter(node_indices, sorted_min_values, label='Minimum', marker='o')
    plt.scatter(node_indices, data_matrix, label='Value', marker='x')
    #plt.scatter(node_indices, sorted_max_values, label='Maximum', marker='^')
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.title(f'Values with Node Degree {model_name} {feat_name} (Ordered by Degree)')
    plt.legend()
    plt.subplot(2, 1, 1)
    plt.scatter(node_indices, degrees, label='Degree', marker='x', color='black')  # Scatter plot for degree
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.title(f'Values with Node Degree {model_name}  {feat_name} (Ordered by Degree)')
    plt.legend()
    #plt.yscale('log', base=2)
    plt.xticks(node_indices)
    
    plt.savefig(f'{model_name}_{feat_name}.png')
    plt.close()

    print("loss of the samples : ", loss / num_graphs)
    
from test_stanford_networks import run_real_networks

def main(seed=1):
    samples = 10
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    set_random_seed(seed)
    dataset = GraphDataset(device=device)
    dataset2 = GraphDataset(device=device)
    dataset3 = GraphDataset(device=device)
    dataset2.load2("MUTAG")
    dataset.load(dataset_path)
    dataset3.load("../data_folder/test")
    getattr(dataset, f'add_{feat_type}')(k)
    getattr(dataset2, f'add_{feat_type}')(k)
    getattr(dataset3, f'add_{feat_type}')(k)
    test_loader2 = GraphDataLoader(dataset2, batch_size=batch_size, shuffle=False)
    test_loader3 = GraphDataLoader(dataset3, batch_size=batch_size, shuffle=False)
    num_training = int(len(dataset) * 0.9)
    num_val = int(len(dataset) * 0.)
    num_test = len(dataset) - num_val - num_training
    generator = torch.Generator().manual_seed(seed)
    train_set, _, test_set = random_split(dataset, [num_training, num_val, num_test], generator=generator)

    train_loader = GraphDataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size, shuffle=False)

    test_loader_samples = GraphDataLoader(test_set, batch_size=samples, shuffle=True)
    test_loader2_samples = GraphDataLoader(dataset2, batch_size=samples, shuffle=True)

    test_graph = GraphDataLoader(dataset, batch_size=1, shuffle=False)

    # normalizer = GraphFeatureNormalizer()
    # normalizer.fit_transform(train_loader)
    # normalizer.transform(test_loader)
    # normalizer.transform(test_loader2)
    
    # Step 2: Create model =================================================================== #
    num_feature, num_classes = k, 1
    set_random_seed(seed)
    #in_dim, hidden_dim, out_dim, num_layers):
    model = GNN(
        1,
        1,
        1,
        1
    ).to(device)

    # Step 3: Create training components ===================================================== #
    
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)  # Replace `parameters` with your specific parameters
  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Step 4: training epoches =============================================================== #
    train_times = []
    train_loss = 0
    for e in range(epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader)
        scheduler.step()
        train_times.append(time() - s_time)

        if (e + 1) % 10 == 0:
            log_format = ("Epoch {}: loss={:f}")
            print(log_format.format(e + 1, train_loss))
   
    test_acc = test_regression(model, test_loader)
    test_acc2 = test_regression(model, test_loader2)
    test_acc3 = test_regression(model, test_loader3)
    print(f"validation : {test_acc}, MUTAG {test_acc2}, Muedim test {test_acc3}")

    real_loss = run_real_networks(model)
    #print_epsilon(model)
    # graph_loss = test_regression2(model, test_graph)
    # print(f'graph loss : {graph_loss}')
    # graph = dataset[0][0]
    # print("label", dataset[0][1])
    # # Print the number of edges and nodes
    # print("Number of edges:", graph.number_of_edges())
    # print("Number of nodes:", graph.number_of_nodes())

    # # Compute the degree of each node
    # degrees = graph.in_degrees()

    # # Print the degrees as a list in one line
    # print("Degrees:", degrees.tolist())




    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         print(f'Weight shape for {name}: {param.shape}')
    #         print(f'Weights for {name}: {param}')
    #     elif 'bias' in name:
    #         print(f'Bias shape for {name}: {param.shape}')
    #         print(f'Biases for {name}: {param}')

    # test_regression_sample(model, test_loader_samples, samples)

    return train_loss, test_acc, test_acc2, test_acc3, real_loss

if __name__ == "__main__":
    x1 = x2 =  x3 = x4 = x5 = 0
    trials = 5
    for i in range(trials):
        train_loss, test_acc, test_acc2, test_acc3, test_acc4 = main(i)
        x1 += train_loss
        x2 += test_acc
        x3 += test_acc2
        x4 += test_acc3
        x5 += test_acc4
    
    print(x1/trials,"    ",x2/trials,"    ",x3/trials,"    ",x4/trials, "    ",x5/trials)
