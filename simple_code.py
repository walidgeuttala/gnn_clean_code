import os
from time import time

import dgl
import torch
import torch.nn as nn
import torch.nn as F
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader

from network import get_network
from utils import set_random_seed, GraphFeatureNormalizer

import os
import torch
import dgl 
from dgl.data import DGLDataset
from dgl import load_graphs
from dgl.data.utils import load_info
from identity import compute_identity

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

    def statistics(self):
        return self.dim_nfeats, self.gclasses, self.device

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
        self.choose_labels(data_path+'/properties_labels.pt')
        self.add_self_loop()
        if self.device == 'cuda':
            self.graphs = [g.to(self.device) for g in self.graphs]
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
        self.choose_labels(f"../data_folder/dgl_graph_labels/{data_name}_properties_labels.pt")
        self.add_self_loop()
        if self.device == 'cuda':
            self.graphs = [g.to(self.device) for g in self.graphs]
            self.labels = self.labels.to(self.device)
        
        

    def has_cache(self):
        '''
        Checks if the processed data has been saved to disk as .bin and .pkl files.
        '''
        # Check if the .bin and .pkl files for the processed data exist in the directory
        graph_path = os.path.join(f'{self.data_path}/dgl_graph.bin')
        info_path = os.path.join(f'{self.data_path}/info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def add_self_loop(self):
        for graph in self.graphs:
            graph = graph.add_self_loop()
    
    def choose_labels(self, file_path):
        # started from 1 as the first labels is the original label
        self.labels = torch.load(file_path)
        self.labels = self.labels[-1].view(-1, 1).float()
        print("check the match ")
        print(self.labels[-10])
        for i in range(10):
            print(self.graphs[i].in_degrees().float().mean().item())
        print('end')

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
            degrees = g.in_degrees().unsqueeze(1).float().to(self.device)
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
    
class GIN(nn.Module):
    def __init__(self, in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers
                 ):

        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_dim, hidden_dim, hidden_dim)
            elif layer == num_layers-1:
                mlp = MLP(hidden_dim, hidden_dim, out_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=True, activation=relu)
            )  # set to True if learning epsilon
            if layer != num_layers-1:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))

        self.pool = (
            AvgPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
        self.relu = nn.ReLU()

    def forward(self, g):
        # list of hidden representation at each layer (including the input layer)
        h = g.ndata["feat"]
        for idx in range(len(self.ginlayers)):
            h = self.ginlayers[idx](g, h)
            h = self.batch_norms[idx](h)
        pooled_h = self.pool(g, h)
        return  pooled_h


def train(model: torch.nn.Module, optimizer, trainloader, args):
    model.train()
    total_loss = 0.0
    num_graphs = 0
    
    loss_func = getattr(F, args.loss_name)(reduction="sum")
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        num_graphs += args.batch_size
    
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
        num_graphs += args.batch_size
        out = model(batch_graphs, args)
        loss += loss_func(out, batch_labels).item()
        args.current_batch += 1

    return loss / num_graphs

def main(seed):
    device = "cuda"
    dataset_path  = ""
    k = 1
    feat_type = ""
    batch_size = 100
    optimizer_name = ""
    lr = 1
    weight_decay = ""
    epochs = 100
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    set_random_seed(seed)
    dataset = GraphDataset(device=device)
    dataset2 = GraphDataset(device=device)
    dataset.load2("MUTAG")
    dataset2.load(dataset_path)
    getattr(dataset, f'add_{feat_type}')(k)
    getattr(dataset2, f'add_{feat_type}')(k)
    
    test_loader2 = GraphDataLoader(dataset2, batch_size=batch_size, shuffle=False)
    num_training = int(len(dataset) * 0.9)
    num_val = int(len(dataset) * 0.)
    num_test = len(dataset) - num_val - num_training
    generator = torch.Generator().manual_seed(seed)
    train_set, _, test_set = random_split(dataset, [num_training, num_val, num_test], generator=generator)

    train_loader = GraphDataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size, shuffle=False)

    normalizer = GraphFeatureNormalizer()
    normalizer.fit_transform(train_loader)
    normalizer.transform(test_loader)
    normalizer.transform(test_loader2)
    
    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()
    hidden_dim = 8
    num_layers = 2

    set_random_seed(seed)
    
    model = GIN(
        in_dim=num_feature,
        hidden_dim=hidden_dim,
        out_dim=num_classes,
        num_layers=num_layers,
    ).to(device)

    # Step 3: Create training components ===================================================== #
    
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)  # Replace `parameters` with your specific parameters
  
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Step 4: training epoches =============================================================== #
    train_times = []
    for e in range(epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader, args)
        scheduler.step()
        train_times.append(time() - s_time)

        if (e + 1) % 10 == 0:
            log_format = ("Epoch {}: loss={:.4f}")
            print(log_format.format(e + 1, train_loss))
   
    test_acc = test_regression(model, test_loader, args)
    test_acc2 = test_regression(model, test_loader2, args)

    print(f"small_test : {test_acc}, medium_test {test_acc2}")
   
    return test_acc, test_acc2, sum(train_times) / len(train_times)

if __name__ == "__main__":
    main()
