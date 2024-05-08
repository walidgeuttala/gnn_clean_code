import os
import torch
import dgl 
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
from identity import compute_identity
import networkx as nx
from utils import calculate_kurtosis_from_degree_list, calculate_avg_shortest_path
import math
from sklearn.preprocessing import StandardScaler

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
    