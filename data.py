import os
import torch
import dgl 
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
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
        if labels != None:
          self.dim_nfeats = len(self.graphs[0].ndata)
          self.gclasses = len(self.labels.unique())
          if self.device == 'cuda':
            self.graphs = [g.to(self.device) for g in self.graphs]
            self.labels = self.labels.to(self.device)
        
    def __len__(self):
        '''
        Returns:
        int: the length of the dataset
        '''
        return len(self.labels)
    
    def process(self):
        '''
        Processes the raw data into a form that is ready for use in machine learning models.
        In this case, no processing is required.
        '''
        pass
    
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

    def save(self, data_path):
        '''
        Saves the processed data to disk as .bin and .pkl files. The processed data consists of the graph data and the corresponding labels.
        '''
        if self.device == 'cuda':
            self.graphs = [g.to("cpu") for g in self.graphs]
            self.labels = self.labels.to("cpu")

        # Save graphs and labels to disk in a .bin file
        graph_path = os.path.join('{}/dgl_graph.bin'.format(data_path))
        save_graphs(graph_path, self.graphs, {'labels':self.labels})
        # Save other information about the dataset in a .pkl file
        info_path = os.path.join('{}/info.pkl'.format(data_path))
        save_info(info_path, {'gclasses': self.gclasses, 'dim_nfeats': self.dim_nfeats, 'device': self.device})

    def load(self, data_path, args):
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
        self.choose_labels(args, data_path+'/properties_labels.pt')
        getattr(self, f'add_{args.feat_type}')(args.k)
        if self.device == 'cuda':
            print('hello')
            self.graphs = [g.to(self.device) for g in self.graphs]
            self.labels = self.labels.to(self.device)
            
        
        

    def load2(self, data_path):
        '''
        Loads the processed data from disk as .bin and .pkl files. The processed data consists of the graph data and the corresponding labels.
        '''
        # Load the graph data and labels from the .bin file
        graph_path = os.path.join('{}/dgl_graph.bin'.format(data_path))
        self.graphs, _ = load_graphs(graph_path)
        
        

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

    def compute_all_labels(self, path):
        results = []
        results.append(self.labels)
        for prop in self.properties:
            for data_type in self.data_types:
                getattr(self, prop)(data_type)
                results.append(self.labels)
        torch.save(results, path) 
    
    def choose_labels(self, args, file_path):
        # started from 1 as the first labels is the original label
        self.labels = torch.load(file_path)
        if args.label_type == 'original':
            self.labels = self.labels[0]
            self.gclasses = 8
            return
        cnt = 1
        if args.data_type == 'regression':
            self.gclasses = 1
        else:
            self.gclasses = 2
        for prop in self.properties:
            for data_type in self.data_types:
                if prop == args.label_type+'_labels' and data_type == args.data_type:
                    self.labels = self.labels[cnt]
                    if args.data_type == 'regression':
                        self.labels =  self.labels.view(-1, 1).float()
                    return 
                
                cnt += 1

    def check_identical_not_same_ref(self, tensor1, tensor2):
        # Check if tensors have identical values
        identical_values = torch.allclose(tensor1, tensor2)

        # Check if tensors have different memory addresses
        same_reference = tensor1.data_ptr() == tensor2.data_ptr()

        return identical_values and not same_reference
    
    def stats_labels(self, args):
        getattr(self, f'{args.label_type}_labels')(args.data_type)
        if args.data_type == 'regression':
            self.labels =  self.labels.view(-1, 1).float()
            #self.labels = self.normalize_labels(self.labels)
        else:
            self.labels =  self.labels.long()

    def original_labels(self, _):
        pass
    
    def normalize_labels(self, labels):
        """
        Normalize labels (targets) using mean and standard deviation.
        
        Args:
            labels (torch.Tensor): Tensor containing the labels to be normalized.
        
        Returns:
            torch.Tensor: Normalized labels.
        """
        # Step 1: Compute Mean and Standard Deviation
        mean = torch.mean(labels)
        std = torch.std(labels)
        
        # Step 2: Normalize Labels
        normalized_labels = (labels - mean) / std
        
        return normalized_labels
    
    def transitivity_labels(self, data_type):
        
        labels = []
        for graph in self.graphs:
            g = graph.cpu().to_networkx()
            g = nx.Graph(g)
            if data_type == "regression":
                self.gclasses = 1
                labels.append(nx.transitivity(g))
            else:
                self.gclasses = 2
                density = nx.density(g)
                labels.append(nx.transitivity(g) > 10*density)
        self.labels = torch.tensor(labels)

    def average_path_labels(self, data_type):
        labels = []
        for graph in self.graphs:
            g = graph.cpu().to_networkx()
            g = nx.Graph(g)
            if data_type == "regression":
                self.gclasses = 1
                labels.append(calculate_avg_shortest_path(graph))
            else:
                self.gclasses = 2
                n = graph.number_of_nodes()
                labels.append(calculate_avg_shortest_path(graph) > math.log2(n))
        self.labels = torch.tensor(labels)

    def density_labels(self, data_type):
        labels = []
        for graph in self.graphs:
            g = graph.cpu().to_networkx()
            g = nx.Graph(g)
            degrees = list(dict(g.degree()).values())
            if data_type == "regression":
                self.gclasses = 1
                labels.append(sum(degrees) / len(degrees))
            else:
                self.gclasses = 2
                labels.append(sum(degrees) / len(degrees)>6)
        self.labels = torch.tensor(labels)
        

    def kurtosis_labels(self, data_type):
        labels = []
        for graph in self.graphs:
            g = graph.cpu().to_networkx()
            g = nx.Graph(g)
            degrees = list(dict(g.degree()).values())
            if data_type == "regression":
                self.gclasses = 1
                labels.append(calculate_kurtosis_from_degree_list(degrees))
                if labels[-1] == None:
                    labels[-1] = 0.
            else:
                self.gclasses = 2
                labels.append(calculate_kurtosis_from_degree_list(degrees)>3)
        self.labels = torch.tensor(labels)
    
    def add_ones_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            g.ndata['feat'] = torch.ones(g.num_nodes(), k).float()
    def add_noise_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs: 
            g.ndata['feat'] = torch.rand(g.num_nodes(), k).float()
    
    def add_degree_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            degrees = g.in_degrees().unsqueeze(1).float()
            repeated_degrees = degrees.repeat(1, k)  # Repeat degree 'k' times
            g.ndata['feat'] = repeated_degrees

    def add_identity_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            g.ndata['feat'] = compute_identity(torch.stack(g.edges(), dim=0), g.number_of_nodes(), k).float()

    def add_norm_degree_feat(self, k):
        self.dim_nfeats = k
        for g in self.graphs:
            degrees = g.in_degrees().unsqueeze(1).float()
            repeated_degrees = degrees.repeat(1, k) / (g.number_of_nodes() - 1) # Repeat degree 'k' times
            g.ndata['feat'] = repeated_degrees
    