import argparse
import subprocess
import itertools
import random

import numpy as np
import pandas as pd
import networkx as nx
import networkit as nk
import dgl
import torch

from data import GraphDataset
from utils import *


def parse_args():
  parser = argparse.ArgumentParser(description="Creating a synthetic networks dataset.")
  parser.add_argument("--output_folder", type=str, help="Output folder name to save the network dataset")
  parser.add_argument("--num_of_networks", type=int, help="Number of networks for every type int")
  parser.add_argument("--networks_names", type=str, help="Networks names to be included in the dataset or use 'all' to include all networks types")
  parser.add_argument("--networks_names", type=str, help="Networks names to be included in the dataset or use 'all' to include all networks types",
                        action=NetworksNamesAction, nargs="+", choices=['ER_low', 'ER_high', 'WS_low', 'WS_high', 'BA_low', 'BA_high', 'grid_low', 'grid_high']) # Add your network choices here
  parser.add_argument("--device", type=str, help="Device the dataset to be saved in 'cuda' or 'cpu'", choices=["cuda", "cpu"])
    
  return parser.parse_args()

class NetworksNamesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values == "all":
            setattr(namespace, self.dest, ['ER_low', 'ER_high', 'WS_low', 'WS_high', 'BA_low', 'BA_high', 'grid_low', 'grid_high']) # Add your network choices here
        else:
            setattr(namespace, self.dest, values)

def generate_parameters(data_dist = [250] * 5, networks="all", degree=4, min_n=250, max_n=1024, seed=42):
  """
  Generate the parameters for the synthetic networks

  Parameters:
    data_dist: number of networks for each network
    networks: if "all", will generate all the networks
    degree: the average degree of the network
    min_n: minimum number of nodes in a network
    max_n: maximum number of nodes in a network

  Returns:
    parameters: dictionary containing all parameters for generating the synthetic networks
  """

  # Dictionary to save the parameters generated for our data generating function
  parameters = dict()
  idx = 0

  # Graph WS parameters
  # Each node is joined with its k nearest neighbors in a ring topology
  min_k = degree
  max_k = degree
  # The probability of rewiring each edge
  min_w = 0.1
  max_w = 0.11
  # Graph BA parameters
  # Number of edges to attach from a new node to existing nodes
  min_m = 4
  max_m = 5

  # Graph HB parameters
  # degree 
  min_degree = 5
  max_degree = 10

  # Graph ER parameters
  seed = np.int64(seed)
  np.random.seed(seed)
  # Generates an array of random integers between min_n and max_n with the size of data_dist[0]
  saved_seed = seed
  np.random.seed(saved_seed)
  n = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
  n = np.array([generate_factors(nn, min_n, max_n) for nn in n])
  x, y = n[:, 0], n[:, 1]
  n = x * y
  #Probability for edge creation.
  def my_random_function(x):
    return np.random.uniform(2/x, 4/x, 1)
    
  # Store the parameters in a 2D numpy array
  if networks == "all" or "ER_low" in networks:
    p = np.vectorize(my_random_function)(n)
    parameters['ER_low'] = np.column_stack((n, p))
    idx += 1

  if networks == "all" or "ER_high" in networks:
    p = np.vectorize(my_random_function)(n)
    parameters['ER_high'] = np.column_stack((n, p))
    idx += 1

  # Graph WS parameters
  # Store the parameters in a 2D numpy array
  if networks == "all" or "WS_low" in networks:
    np.random.seed(seed)
    # Generates an array of random integers between min_k and max_k even with the size of data_dist[idx]
    k = np.array(np.random.randint(min_k, max_k+1, data_dist[idx]))
    k = np.floor_divide(k, 2) * 2
    seed += 1
    np.random.seed(seed)
    # Generates an array of random floating point numbers between min_w and max_w with the size of data_dist[idx]
    p = np.random.uniform(min_w, max_w, data_dist[idx])
    seed+=1
    parameters['WS_low'] = np.column_stack((n, k, p))
    idx += 1

  if networks == "all" or "WS_high" in networks:
    np.random.seed(seed)
    # Generates an array of random integers between min_k and max_k even with the size of data_dist[idx]
    k = np.array(np.random.randint(min_k, max_k+1, data_dist[idx]))
    k = np.floor_divide(k, 2) * 2 * 2
    seed += 1
    np.random.seed(seed)
    # Generates an array of random floating point numbers between min_w and max_w with the size of data_dist[idx]
    p = np.random.uniform(min_w, max_w, data_dist[idx])
    seed+=1
    parameters['WS_high'] = np.column_stack((n, k, p))
    idx += 1

  # graph BA parameters
  # Store the parameters for the BA graph in a dictionary
  if networks == "all" or "BA_low" in networks:
    np.random.seed(seed)
    m = np.array(np.random.randint(min_m, max_m, data_dist[idx]))
    seed += 1
    parameters['BA_low'] = np.column_stack((n, m))
    idx += 1

  if networks == "all" or "BA_high" in networks:
    np.random.seed(seed)
    m = np.array(np.random.randint(min_m, max_m, data_dist[idx]))
    seed += 1
    parameters['BA_high'] = np.column_stack((n, m))
    idx += 1

  # graph GRID low transtivity parameters
  # Store the parameters for the grid_low graph in a dictionary
  if networks == "all" or "grid_low" in networks:
    parameters['grid_tr_low'] = np.column_stack((x, y))
    idx += 1

  # graph GRID high transtivity parameters
  # Store the parameters for the grid_high graph in a dictionary
  if networks == "all" or "grid_high" in networks:
    parameters['grid_tr_high'] = np.column_stack((x, y))
    idx += 1
  
  # graph hyperbolic graph parameters
  # Store the parameters for the hyperbolic graph in a dictionary
  if networks == "all" or "HB" in networks:
    np.random.seed(saved_seed)
    x = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
    seed += 1
    np.random.seed(seed)
    y = np.array(np.random.randint(min_degree, max_degree, data_dist[idx]))
    seed += 1
    parameters["HB"] = np.column_stack((x, y))
    idx += 1
  
  if networks == "all" or "PC" in networks:
    np.random.seed(saved_seed)
   
    seed += 1

    np.random.seed(seed)
    m = np.array(np.random.randint(min_m, max_m, data_dist[idx]))
    seed += 1
    parameters["PC"] = np.column_stack((n, m))
    idx += 1

  # Reset the seed for random number generation
  np.random.seed(None)

  # Return the graph parameters as a dictionary
  return parameters

def generate_data(param, data_dist, degree_fixed = 4, networks="all"):
    '''
    generate_data: This function generates data for 8 types of graphs given the parameters

    Parameters:
        - param: Dictionary that contains the parameters for each type of graph
        - data_dist: List that contains the count of each type of graph to generate
        - networks: all, or choose the networks you want
    Returns:
        - graphs: List of generated graphs
        - classes: List of 0s, 1s, 2s, and so on indicating the class of each graph
    '''
    graphs = []
    idx = 0
    
    # ER Graphs
    if networks == "all" or "ER_low" in networks:
      for i in range(data_dist[idx]):
          graphs.append(nx.gnp_random_graph(int(param['ER_low'][i, 0]), degree_fixed / (int(param['ER_low'][i, 0])), seed=i, directed=False))
      idx += 1

    if networks == "all" or "ER_high" in networks:
      for i in range(data_dist[idx]):
          graphs.append(nx.gnp_random_graph(int(param['ER_high'][i, 0]), degree_fixed*2 / (int(param['ER_high'][i, 0])), seed=i, directed=False))
      idx += 1
    
    # WS Graphs
    if networks == "all" or "WS_low" in networks:
      for i in range(data_dist[idx]):
          edges = int(param['WS_low'][i,0]*param['WS_low'][i,1])
          graphs.append(nx.watts_strogatz_graph(int(param['WS_low'][i,0]), int(param['WS_low'][i,1]), param['WS_low'][i,2], seed=i))
      idx += 1

    if networks == "all" or "WS_high" in networks:
      for i in range(data_dist[idx]):
          edges = int(param['WS_high'][i,0]*param['WS_high'][i,1])
          graphs.append(nx.watts_strogatz_graph(int(param['WS_high'][i,0]), int(param['WS_high'][i,1]), param['WS_high'][i,2], seed=i))
      idx += 1
    
    # BA Graphs
    if networks == "all" or "BA_low" in networks:
      for i in range(data_dist[idx]):
          graphs.append(nx.barabasi_albert_graph(int(param['BA_low'][i,0]), min(degree_fixed // 2, int(param['BA_low'][i,0]) - 1), seed=i, initial_graph=None))
      idx += 1

    if networks == "all" or "BA_high" in networks:
      for i in range(data_dist[idx]):
          graphs.append(nx.barabasi_albert_graph(int(param['BA_high'][i,0]), min(degree_fixed*2 // 2, int(param['BA_high'][i,0]) - 1), seed=i, initial_graph=None))
      idx += 1
    
    # 2D Grid using manhattan distance low transitivity
    if networks == "all" or "grid_low" in networks:
      for i in range(data_dist[idx]):
        graphs.append(create_manhattan_2d_grid_graph(int(param['grid_tr_low'][i, 0]), int(param['grid_tr_low'][i, 1]), 1))
      idx += 1
    
    # 2D Grid using moore distance high transitivity
    if networks == "all" or "grid_high" in networks:
      for i in range(data_dist[idx]):
        graphs.append(create_moore_2d_grid_graph(int(param['grid_tr_high'][i, 0]), int(param['grid_tr_high'][i, 1]), 1))
      idx += 1

    # Resetting the seed for numpy
    np.random.seed(None)

    # Return the graphs list and the labels list
    return graphs, torch.LongTensor([i for i, x in enumerate(data_dist) for _ in range(x)])

def generate_combinations(choices):
    for combo in itertools.product(*[range(choice) for choice in choices]):
        yield combo

def create_moore_2d_grid_graph(n, m, r):
    if r == 0:
        return nx.Graph()
    G = nx.Graph()
    
    combinations_generator = generate_combinations([n, m])
    for combo in combinations_generator:
        G.add_node(combo)
    
    for node in G.nodes():
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if i != 0 or j != 0:
                    node2 = ((node[0] + i + n) % n, (node[1] + j + m) % m )
                    if node == node2:
                        continue
                    G.add_edge(node, node2)
  
    return G

def create_manhattan_2d_grid_graph(n, m, r):
    if r == 0:
        return nx.Graph()
    G = nx.Graph()
    
    combinations_generator = generate_combinations([n, m])
    for combo in combinations_generator:
        G.add_node(combo)
    
    for node in G.nodes():
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if 0 < abs(i) + abs(j) <= r:
                    node2 = ((node[0] + i + n) % n, (node[1] + j + m) % m )
                    if node == node2:
                        continue
                    G.add_edge(node, node2)

    return G

def create_DF(param, graphs, data_dist):
  length = sum(data_dist) // len(param)
  dataFrames = []

  # Add data to the list `edges`
  edges = dict()
  nodes = dict()

  idx = 0
  for i, key in enumerate(param.keys()):
    # Append the number of edges for a type of graph to `edges`
    edges[key] = np.array([g.number_of_edges() for g in graphs[idx:idx+data_dist[i]]])
    # Increase the index by the number of graphs of this type
    idx += data_dist[i]

  idx = 0
  for i, key in enumerate(param.keys()):
    # Append the number of edges for a type of graph to `edges`
    nodes[key] = np.array([g.number_of_nodes() for g in graphs[idx:idx+data_dist[i]]])
    # Increase the index by the number of graphs of this type
    idx += data_dist[i]

  for key in param.keys():
    dataFrames.append(pd.DataFrame({'Num_nodes': nodes[key], 'Num_edges':edges[key]}))


  # Concatenate the dataframes along axis 1 (i.e., horizontally)
  df = pd.concat(dataFrames, axis=1)

  # Create a hierarchical column index with the top level being the column names of the original dataframes
  df.columns = pd.MultiIndex.from_product([param.keys(), ['Num_nodes', 'Num_edges']])


  for i, key in enumerate(param):
    ave_degree = df[key]['Num_edges'] * 2/df[key]['Num_nodes']
    df.insert(i+i*2+2, column=(key, 'Average_degree'), value=ave_degree)

  for i, key in enumerate(param):
    density = []
    for j in range(length):
      density.append(nx.density(graphs[i*length+j]))
    df.insert(i+i*3+3, column=(key, 'Density'), value=density)

  for i, key in enumerate(param):
    transitivity = []
    for j in range(length):
      transitivity.append(nx.transitivity(graphs[i*length+j]))
    df.insert(i+i*4+4, column=(key, 'Transitivity'), value=transitivity)

  for i, key in enumerate(param):
    avg_shortest_path = []
    for j in range(length):
      avg_shortest_path.append(calculate_avg_shortest_path(dgl.from_networkx(graphs[i*length+j])))
    df.insert(i+i*5+5, column=(key, 'avg_shortest_path'), value=avg_shortest_path)  

  for i, key in enumerate(param):
    list_max_degrees = []
    for j in range(length):
      max_degree = np.max(np.array([d for n, d in graphs[i*length+j].degree()]))
      list_max_degrees.append(max_degree)
    df.insert(i+i*6+6, column=(key, 'max_degree'), value=list_max_degrees) 

  for i, key in enumerate(param):
    list_variance_degrees = []
    for j in range(length):
      degree_variance = np.var(np.array([d for n, d in graphs[i*length+j].degree()]))
      list_variance_degrees.append(degree_variance)
    df.insert(i+i*7+7, column=(key, 'variance_degree'), value=list_variance_degrees)
    
  return df
    

def add_summary(df):
  df.to_csv('data/info_about_graphs.csv', index=False)

  summary = df.describe()

  # save the summary statistics to a file
  with open('data/summary.txt', 'w') as f:
      f.write(summary.to_string())

  command = "touch data/README.md"
  subprocess.run(command, shell=True)
  with open('data/README.md', 'w') as f:
    f.write('In the data folder, you will find several components that contain information about the dataset. These components are described below:\n\n1. dgl_graph.bin and info.pkl: These are the two main files that contain the structure of the graphs and additional information about the dataset. The dgl_graph.bin file contains the graph structure, and the info.pkl file contains the number of features and the number of classes for the data. To load these files, you can use the `load()` method from the GraphDataset class in the DGL library.\n\n2. info_about_graphs.csv and summary.txt: These files contain additional information about the graphs in the dataset, including the number of nodes, edges, degree, and density. Note that our graphs are undirected.\n\n3. Box plots: Lastly, you will find three box plots that provide a visual representation of the dataset. These plots can help you to better understand the distribution of the data and identify any potential outliers.\n\n4. parameters_generated_data.pth: This file contains a dictionary of parameters used to generate the data. You can load it using the torch.load() method.\n\nTo use this dataset, you can load the dgl_graph.bin and info.pkl files using the GraphDataset class in the DGL library. You can also refer to the info_about_graphs.csv and summary.txt files to get additional information about the graphs. Finally, you may find it helpful to review the box plots to gain a better understanding of the data distribution.')

if __name__ == "__main__":
  # make sure we don't have the folder by removing it
  args = parse_args()
  create_data_folder(args.output_folder)

  data_dist =  [args.num_of_networks] * len(args.networks_names)
  # Generate parameters for given data distribution
  parameters = generate_parameters(data_dist, args.networks_names)

  # Generate the graphs and corresponding labels
  graphs, labels = generate_data(parameters, data_dist, args.networks_names)

  # Transform networkx graphs into DGL graphs
  graphs = [dgl.from_networkx(graph) for graph in graphs]

  # Create a Graphdataset using the generated graphs and labels
  dataset = GraphDataset(graphs, labels, args.device)

  # Save dataset
  dataset.save(args.output_folder)

  # Saving parameters
  torch.save(parameters, f'{args.output_folder}/parameters_generated_data.pth')