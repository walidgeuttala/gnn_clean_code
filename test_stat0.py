import networkx as nx
import dgl
from data import GraphDataset
import numpy as np
from utils import *
from test_stanford_networks import *
import math
from collections import Counter
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import numpy as np
import pickle

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
                    average_shortest_path.append(calculate_avg_shortest_path(graph))
                    graph = dgl.to_networkx(graph)
                    graph = nx.Graph(graph)
                    names.append(name[:-7])
                    graphs.append(graph)

    for file_path in result:
        graph, name = read_graph(file_path)
        graph2 = dgl.from_networkx(graph)
        average_shortest_path.append(calculate_avg_shortest_path(graph2))
        names.append(name)
        graphs.append(graph)

    for file_path in list_names:
        graph2 = read_graph2(file_path)
        graph = dgl.to_networkx(graph2)
        graph = nx.Graph(graph)
        average_shortest_path.append(calculate_avg_shortest_path(graph2))
        names.append(file_path[-1])
        graphs.append(graph)

    return graphs, names, average_shortest_path

def read_synthtic_graphs(path):
    device = 'cpu'
    dataset = GraphDataset(device=device)
    dataset.load2(path)
    average_shortest_path = []
    graphs = dataset.graphs
    for idx in range(len(graphs)):
        average_shortest_path.append(calculate_avg_shortest_path(graphs[idx]))
        graphs[idx] = graphs[idx]
        graphs[idx] = dgl.to_networkx(graphs[idx])
        graphs[idx] = nx.Graph(graphs[idx])

    return graphs, average_shortest_path

def calculate_kurtosis_from_degree_list(degree_list):
    """
    Calculate the kurtosis for a given list of degrees in a graph.

    Parameters:
    - degree_list (list): List of degrees in the graph.

    Returns:
    - float: Kurtosis value.
    """
    try:
        # Convert the degree list into a degree distribution
        degree_distribution = dict(Counter(degree_list))
        
        # Calculate the kurtosis for the degree distribution
        kurtosis_value = kurtosis(list(degree_distribution.values()))
        
        return kurtosis_value
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def stats0(graphs, average_shortest_path):
    arr = [[] for _ in range(4)]
    arr[0] = average_shortest_path.copy()
    for idx, graph in enumerate(graphs):
        n = nx.number_of_nodes(graph)
        # High shortest path or no
        arr[0][idx] = math.log2(n) < arr[0][idx]
        density = nx.density(graph)
        # high or low transitivity
        arr[1].append(nx.transitivity(graph) > 10*density)
        degrees = list(dict(graph.degree()).values())
        # return True if left skewed scale free 
        # NAn values will result in False which is what I want
        kurtosis = calculate_kurtosis_from_degree_list(degrees)
        if math.isnan(kurtosis):
            kurtosis = 0.
        arr[2].append(kurtosis>3)
        # High density or no 
        arr[3].append(sum(degrees) / len(degrees)>6)
            

    arr2 = [[] for _ in range(4)]
    arr2[0] = average_shortest_path
    for idx, graph in enumerate(graphs):
        n = nx.number_of_nodes(graph)
        # High shortest path or no
        density = nx.density(graph)
        # high or low transitivity
        arr2[1].append(nx.transitivity(graph))
        degrees = list(dict(graph.degree()).values())
        # return True if left skewed scale free 
        kurtosis = calculate_kurtosis_from_degree_list(degrees)
        if math.isnan(kurtosis):
            kurtosis = 0.
        arr2[2].append(kurtosis)
        # High density or no 
        arr2[3].append(sum(degrees) / len(degrees))
             
    return np.array(arr).T, np.array(arr2).T

def compute_density(graphs):
    """
    Compute the density for each graph in the list.

    Parameters:
    - graphs (list): List of NetworkX graphs.

    Returns:
    - list: List of densities corresponding to each graph.
    """
    densities = [nx.density(graph) for graph in graphs]
    return np.array(densities)

def average_degree_list(graphs):
    """
    Compute the average degree for a list of NetworkX graphs.

    Parameters:
    - graphs: List of NetworkX graphs.

    Returns:
    - List of average degrees for each graph.
    """
    average_degrees = []

    for G in graphs:
        # Compute the average degree using average_degree_connectivity
        avg_degree_dict = nx.average_degree_connectivity(G)
        
        # For simplicity, just take the average of the values in the dictionary
        avg_degree = sum(avg_degree_dict.values()) / len(avg_degree_dict)
        
        average_degrees.append(avg_degree)

    return np.array(average_degrees)

def plot_histograms(arr, name, graph_type, samples):
    """
    Plot histograms for densities and save the plots.

    Parameters:
    - densities (list): List of densities.
    """
    names = ['ER_low', 'ER_high', 'WS_low', 'WS_high', 'BA_low', 'BA_high', 'Grid_low', 'Grid_high']
    # Plot histograms for each set of 250 graphs and save them
    idx = 0
    for i in range(0, len(arr), samples):
        plt.figure(figsize=(10, 5))
        plt.hist(arr[i:i+samples], bins=20, color='blue', alpha=0.7)
        plt.title(f'{name} Histogram for {names[idx]} {graph_type} Graphs')
        plt.xlabel(f'{name}')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'{name}_histogram_{names[idx]}_{graph_type}.png')  # Save the plot
        plt.close()
        idx += 1

    # Plot a general histogram for all 2000 graphs and save it
    plt.figure(figsize=(10, 5))
    plt.hist(arr, bins=40, color='green', alpha=0.7)
    plt.title(f'Overall {name} Histogram for All {graph_type} Graphs')
    plt.xlabel(f'{name}')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'overall_{name}_{graph_type}_histogram.png')  # Save the plot
    plt.close()

def count_synthtic(arr, samples):
    arr = arr.reshape(-1, samples, arr.shape[1])
    return np.sum(arr, axis=1)

def average_synthtic(arr, samples):
    arr = arr.reshape(-1, samples, arr.shape[1])
    return np.mean(arr, axis=1)

def read_gem_ben_graphs():
    graphs = []
    average_shortest_path = []
    average_degree = []
    for i in range(100):
        # Specify the file path to your graph.gpickle file
        file_path = "real_graphs/{}/graph.gpickle".format(i)
        
        # Read the graph from the gpickle file
        with open(file_path, "rb") as file:
            graph = pickle.load(file)
        
        # Now you can use the 'graph' variable to work with your loaded graph
        graph = nx.Graph(graph)
        #graph = graph.to_undirected()
        graphs.append(graph)
        graph2 = dgl.from_networkx(graph)
        average_shortest_path.append(calculate_avg_shortest_path(graph2))
        average_degree.append(sum(dict(graph.degree()).values()) / len(graph))
    df = pd.read_hdf("real_graphs_list_100.h5", 'df')

    return graphs, df, average_shortest_path, average_degree
    


#graphs, df, average_shortest_path,average_degree = read_gem_ben_graphs()
#print(average_shortest_path)

#stat, arr = stats0(graphs, average_shortest_path)

#names = ['average_shortest_path', 'transtivity', 'kurtosis', 'density', 'edges']

#for i in range(5):
#    df[names[i]+'_value'] = stat[:, i]
#    df[names[i]] = arr[:, i]
#df['avg_degree2'] = np.array(average_degree)
#df.to_csv('gemben.csv', index=False)    
#print(df[names])
#print(stat)
#print(arr)

# result = download_and_extract(linkss)
# graphs, names, average_shortest_path = read_real_graphs(result)
# subnames = ['facebook_combined', 'wiki-Vote', 'p2p-Gnutella04', 'p2p-Gnutella08', 'CSphd', 'geom', 'netsience', 'adjnoun', 'football', 'hep-th', 'netscience', 'CLUSTERDataset', 'TreeGridDataset']
# subname_indices = [idx for idx, name in enumerate(names) if name in subnames]
# graphs = [graphs[idx] for idx in subname_indices]
# average_shortest_path = [average_shortest_path[idx] for idx in subname_indices]

# stat1, arr2 = stats0(graphs, average_shortest_path)

# self.properties = ['transitivity_labels', 'average_path_labels', 'density_labels', 'kurtosis_labels']
#         self.data_types = ['regression', 'classification']
graphs, average_shortest_path = read_synthtic_graphs('../data_folder/data')
stat2, arr2 = stats0(graphs, average_shortest_path)
result = list()
result.append(torch.tensor(np.repeat(np.arange(8), 250)).view(-1, 1).to(torch.int32))
for i in range(4):
    result.append(torch.tensor(stat2[:, i]).view(-1, 1).to(torch.float32))
    result.append(torch.tensor(arr2[:, i]).view(-1, 1).to(torch.float32))
    print(torch.sum(torch.isnan(result[-2])))
    print(torch.sum(torch.isnan(result[-1])))
torch.save(result, "../data_folder/data/properties_labels.pt")

graphs, average_shortest_path = read_synthtic_graphs('../data_folder/test')
stat2, arr2 = stats0(graphs, average_shortest_path)
result = list()
result.append(torch.tensor(np.repeat(np.arange(8), 50)).view(-1, 1).to(torch.float32))
for i in range(4):
    result.append(torch.tensor(stat2[:, i]).view(-1, 1).to(torch.float32))
    result.append(torch.tensor(arr2[:, i]).view(-1, 1).to(torch.float32))
    print(torch.sum(torch.isnan(result[-2])))
    print(torch.sum(torch.isnan(result[-1])))
torch.save(result, "../data_folder/test/properties_labels.pt")



# for i in range (7, 8):
#     print(np.array(labels[i].view(-1, 1).float())[-10:])
# #print(stat2.shape())


# stat2 = count_synthtic(stat2, 250)
# arr3 = average_synthtic(arr3, 250)

# print(stat1)
# #print(stat2)
# print(arr2)
#print(arr3)

#degrees = average_degree_list(graphs)
#plot_histograms(degrees, 'Average Degree', 'Small Graphs', 250)

#graphs, average_shortest_path = read_synthtic_graphs('test')
#plot_histograms(degrees, 'Average Degree', 'Medium Graphs', 50)
#stat2, arr3 = stats0(graphs, average_shortest_path)
#stat2 = count_synthtic(stat2, 50)
#arr3 = average_synthtic(arr3, 50)

#print(stat2)
#print(arr3)



# Compute densities
#densities = compute_density(graphs)

#print(np.mean(densities))

# Plot histograms and save them
#plot_density_histograms(densities)
