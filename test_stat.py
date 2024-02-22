from scipy.stats import skew
import networkx as nx
import dgl
from data import GraphDataset
import numpy as np
from utils import *
from test_stanford_networks import *
from scipy.stats import entropy
import scipy.interpolate as interp
import random

def generate_random_indexes(x):
    # Generate x random indexes between [0, 250[ uniformly
    random_indexes = random.sample(range(250), x)
    return random_indexes


def divg_count(degree_list_graph1, degree_list_graph2):
    # Convert degree lists into degree distributions
    degree_dist_graph1, bins1 = np.histogram(degree_list_graph1, bins=np.arange(min(degree_list_graph1), max(degree_list_graph1) + 2))
    degree_dist_graph2, bins2 = np.histogram(degree_list_graph2, bins=np.arange(min(degree_list_graph2), max(degree_list_graph2) + 2))

    # Use the union of bin edges to ensure consistency in binning
    bins = np.union1d(bins1, bins2)
    # Normalize degree distributions
    degree_dist_graph1_normalized = degree_dist_graph1 / np.sum(degree_dist_graph1)
    degree_dist_graph2_normalized = degree_dist_graph2 / np.sum(degree_dist_graph2)

    degree_dist_graph1_normalized = np.interp(bins, bins1[:-1], degree_dist_graph1_normalized, left=0, right=0)
    degree_dist_graph2_normalized = np.interp(bins, bins2[:-1], degree_dist_graph2_normalized, left=0, right=0)
    # Ensure both distributions have the same shape
    degree_dist_graph1_normalized = degree_dist_graph1_normalized[:len(bins) - 1]
    degree_dist_graph2_normalized = degree_dist_graph2_normalized[:len(bins) - 1]


    # Calculate probability distributions
    p = 0.5 * (degree_dist_graph1_normalized + degree_dist_graph2_normalized)

    # Compute Jensen-Shannon Divergence
    jsd = 0.5 * (entropy(degree_dist_graph1_normalized, p) + entropy(degree_dist_graph2_normalized, p))

    return jsd

def read_real_graphs(result):
    names = []
    graphs = []
    average_shortest_path = []
    with open('links.txt', 'r') as file:
        f = 0
        for line in file:
            if len(line) == 1:
                f = 1
            else :
                if f == 1:
                    f = 0
                else:
                    graph, name =  download_Stanford_network(line[:-1])
                    average_shortest_path.append(calculate_avg_shortest_path(graph) / (graph.number_of_nodes()-1))
                    graph = dgl.to_networkx(graph)
                    graph = nx.Graph(graph)
                    names.append(name[:-7])
                    graphs.append(graph)
                    

    for file_path in result:
        graph, name = read_graph(file_path)
        graph2 = dgl.from_networkx(graph)
        average_shortest_path.append(calculate_avg_shortest_path(graph2) / (graph2.number_of_nodes()-1))
        names.append(name)
        graphs.append(graph)


    for file_path in list_names:
        graph2 = read_graph2(file_path)
        graph = dgl.to_networkx(graph2)
        graph = nx.Graph(graph)
        average_shortest_path.append(calculate_avg_shortest_path(graph2) / (graph2.number_of_nodes()-1))
        names.append(file_path[-1])
        graphs.append(graph)

    return graphs, names, average_shortest_path

def choose_indexes_with_increment(index_list, increment):
    chosen_indexes = index_list

    # Iterate through each specified index and choose subsequent indexes with increments
    for index in range(2000//250-1):
        index_list = [x + 250 for x in index_list]
        chosen_indexes += index_list 

    return chosen_indexes

def read_synthtic_graphs(idxs):
    device = 'cpu'
    dataset = GraphDataset(device=device)
    dataset.load('data')
    average_shortest_path = []
    idxs = choose_indexes_with_increment(idxs, 250)
    graphs = [dataset.graphs[index] for index in idxs]
    
    for idx in range(len(graphs)):
        average_shortest_path.append(calculate_avg_shortest_path(graphs[idx]) / (graphs[idx].number_of_nodes()-1))
        graphs[idx] = graphs[idx]
        graphs[idx] = dgl.to_networkx(graphs[idx])
        graphs[idx] = nx.Graph(graphs[idx])
    return graphs, average_shortest_path 

def stats2(graphs, average_shortest_path):
    arr = [[] for _ in range(4)]
    #names = ['Nodes', 'Edges', 'Density', 'Transitivity', 'Average Degree', 'transitivity_density', 'var Degree', 'Average Shortest Path','avg_path_nodes', 'skew']
    names = ['Average Shortest Path Norm', 'TransitivityByDensity', 'Degree Dist', 'AvgClusteringByDensity', 'DiameterNorm']
    arr[0] = average_shortest_path
    for graph in graphs:
        density = nx.density(graph)
        arr[1].append(nx.transitivity(graph) / density)
        degrees = list(dict(graph.degree()).values())
        arr[2].append(degrees)
        arr[3].append(nx.average_clustering(graph) / density)
        #arr[4].append(nx.diameter(graph) / graph.number_of_nodes())
    
    return arr

def compare(stat1, stat2):
    ans_table = [[] for _ in range(len(stat1[0]))]
    #ans_table = np.zeros((len(stat1[0]), len(stat2[0])), dtype=np.float32)
    for i in range(len(stat1[0])):
        for j in range(len(stat2[0])):
            ans = []
            ans.append(abs(stat1[0][i] - stat2[0][j]))
            ans.append(abs(stat1[1][i] - stat2[1][j]))
            ans.append(abs(divg_count(stat1[2][i], stat2[2][j])))
            ans.append(abs(stat1[3][i] - stat2[3][j]))
            
            ans_table[i].append(ans)           
    
    ans_table = np.array(ans_table)
    min_vals = np.min(ans_table, axis=(0, 1), keepdims=True)
    max_vals = np.max(ans_table, axis=(0, 1), keepdims=True)
    ans_table = (ans_table - min_vals) / (max_vals - min_vals)
    ans_table = np.sum(ans_table, axis=-1)
    return ans_table

def sort_rows_and_return_indices(matrix):
    # Get the indices that would sort each row
    sorted_indices = np.argsort(matrix, axis=1)

    # Create an array filled with the sorted indices
    sorted_values = np.take_along_axis(matrix, sorted_indices, axis=1)

    return sorted_values, sorted_indices




samples = 250
#idxs = generate_random_indexes(samples)
idxs = list(range(250))
result = download_and_extract(linkss)
graphs, names, average_shortest_path = read_real_graphs(result)
subnames = ['facebook_combined', 'wiki-Vote', 'p2p-Gnutella04', 'p2p-Gnutella08', 'CSphd', 'geom', 'netsience', 'adjnoun', 'football', 'hep-th', 'netscience', 'CLUSTERDataset', 'TreeGridDataset']
subname_indices = [idx for idx, name in enumerate(names) if name in subnames]
graphs = [graphs[idx] for idx in subname_indices]
average_shortest_path = [average_shortest_path[idx] for idx in subname_indices] 

stat1 = stats2(graphs, average_shortest_path)

graphs, average_shortest_path = read_synthtic_graphs(idxs)
stat2 = stats2(graphs, average_shortest_path)

ans = compare(stat2, stat1)
ans = ans.reshape(-1, samples, ans.shape[1])
# Compute the average along axis 1 (across the 10 samples)
ans = np.mean(ans, axis=1)
ans = ans.T
ans = ans / ans.sum(axis=1, keepdims=True)
print(ans)
sorted_values, sorted_indices = sort_rows_and_return_indices(ans)
#print(sorted_values)
print(sorted_indices)




