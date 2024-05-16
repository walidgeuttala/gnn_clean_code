import argparse
import gzip
import subprocess
import numpy as np
import dgl
from network import get_network
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import os 
import json 
import torch
from identity import compute_identity
from utils import calculate_avg_shortest_path
import logging
import requests
import zipfile
from io import BytesIO
from pathlib import Path
import re
from main import parse_args

list_names = [['CLUSTERDataset'], ['PATTERNDataset'], ['TreeGridDataset'], ['TreeCycleDataset']]

linkss = ['http://vlado.fmf.uni-lj.si/pub/networks/data/GED/CSphd.ZIP', 'http://vlado.fmf.uni-lj.si/pub/networks/data/bio/Yeast/yeast.zip', 'http://vlado.fmf.uni-lj.si/pub/networks/data/collab/Geom.zip',
         'http://vlado.fmf.uni-lj.si/pub/networks/data/collab/NetScience.zip', 'http://vlado.fmf.uni-lj.si/pub/networks/data/Erdos/Erdos02.net',
         'http://www-personal.umich.edu/~mejn/netdata/karate.zip', 'http://www-personal.umich.edu/~mejn/netdata/lesmis.zip', 'http://www-personal.umich.edu/~mejn/netdata/adjnoun.zip',
         'http://www-personal.umich.edu/~mejn/netdata/football.zip', 'http://www-personal.umich.edu/~mejn/netdata/dolphins.zip',
         'http://www-personal.umich.edu/~mejn/netdata/polbooks.zip',
         'http://www-personal.umich.edu/~mejn/netdata/hep-th.zip',  'http://www-personal.umich.edu/~mejn/netdata/power.zip']

def parse_args2():
    parser = argparse.ArgumentParser(description="Testing Stanford Networks")
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="dataset",
        help="path for weights of the model",
    )
    parser.add_argument(
        "--args_file",
        type=str,
        default=False,
        help="args json file for the model that we trained with",
    )
    parser.add_argument(
        "--real_network_folder",
        type=str,
        default="../data_folder/",
        help="args json file for the model that we trained with",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="args json file for the model that we trained with",
    )
    parser.add_argument(
        "--dist_draw",
        type=bool,
        default=False,
        help="dist draw for networks list",
    )
    parser.add_argument(
        "--link_data",
        type=str,
        default="../data_folder/",
        help="args json file for the model that we trained with",
    )
   

    return parser.parse_args()
def make_graph_bidirectional(graph):
    src, dst = graph.edges()
    reversed_edges = (dst, src)  # Reverse the edges
    new_src, new_dst = graph.edges() + reversed_edges
    return dgl.graph((new_src, new_dst))

def read_graph2(name):
    #name.insert(0, 'data')
    obj = dgl.data
    for method_name in name:
        obj = getattr(obj, method_name)
    graph = obj()
    if isinstance(graph, list):
        print(len(graph))
    graph = graph[0]
    # Convert to NetworkX graph
    #graph = dgl.to_homogeneous(graph)
    
    graph = dgl.to_homogeneous(graph)
    #graph = make_graph_bidirectional(graph)
    #nx_graph = graph.to_networkx().to_undirected()
   
    return graph

def stanford_degree_dist_plots(result, draw = True):
    # Open the file in read mode
    names = []
    nodes = []
    edges = []
    density = []
    transitivity = []
    trans_dens = []
    avg_short_path = []
    avg_degree = []
    var_degrees = []
    avg_path_nodes = []
    transitivity_density = []
    skewed = []
    from scipy.stats import skew
    with open('../links.txt', 'r') as file:
        # Iterate over each line in the file
        f = 0
        for line in file:
            # Print the line
            
            if len(line) == 1:
                f = 1
            else :
                if f == 1:
                    f = 0
                else:
                    graph, name =  download_Stanford_network(line[:-1], )
                    avg_short_path.append(calculate_avg_shortest_path(graph))
                    graph = dgl.to_networkx(graph)
                    graph = nx.Graph(graph)
                    names.append(name[:-7])
                    nodes.append(graph.number_of_nodes())
                    edges.append(graph.number_of_edges())
                    avg_degree.append(edges[-1]*2/nodes[-1])
                    density.append(nx.density(graph))
                    transitivity.append(nx.transitivity(graph))
                    transitivity_density.append(transitivity[-1]>density[-1])
                    degrees = list(dict(graph.degree()).values())
                    var_degrees.append(np.var(degrees))
                    avg_path_nodes.append(avg_short_path[-1]<np.log(nodes[-1]))
                    skewed.append(skew(degrees))
                    trans_dens.append(transitivity[-1]/density[-1])
                    
                    if draw == True:
                        print("network name : "+names[-1])
                        print()
                        print('number of nodes : ',nodes[-1])
                        print('nuler of edges : ', edges[-1])
                        print('average_shortest_path', avg_short_path[-1])
                        print("transitivity : ",transitivity[-1])
                        print("density : ",density[-1])
                        graph = nx.Graph(graph)
                        
                        degrees = [val for (node, val) in graph.degree()]
                        plt.clf()
                        plt.hist(degrees, bins=100)  # Adjust bins as needed
                        plt.xlabel("Degree")
                        plt.ylabel("Frequency")
                        plt.title("Degree Distribution")
                        plt.show()
                        plt.savefig('./degree_dist_{}.png'.format(name))

    for file_path in result:
        graph, name = read_graph(file_path)
        grpah2 = dgl.from_networkx(graph)
        avg_short_path.append(calculate_avg_shortest_path(grpah2))
        names.append(name)
        density.append(nx.density(graph))
        transitivity.append(nx.transitivity(graph))
        nodes.append(graph.number_of_nodes())
        edges.append(graph.number_of_edges())
        trans_dens.append(transitivity[-1]/density[-1])
        avg_degree.append(edges[-1]*2/nodes[-1])
        transitivity_density.append(transitivity[-1]>density[-1])
        degrees = list(dict(graph.degree()).values())
        var_degrees.append(np.var(degrees))
        avg_path_nodes.append(avg_short_path[-1]<np.log(nodes[-1]))
        skewed.append(skew(degrees))
        if draw == True:
            print("network name : "+names[-1])
            print()
            print('number of nodes : ',nodes[-1])
            print('nuler of edges : ', edges[-1])
            print('average_shortest_path', avg_short_path[-1])
            print("transitivity : ",transitivity[-1])
            print("density : ",density[-1])
            graph = nx.Graph(graph)

            degrees = [val for (node, val) in graph.degree()]
            plt.clf()
            plt.hist(degrees, bins=100)  # Adjust bins as needed
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.title("Degree Distribution")
            plt.show()
            plt.savefig('./degree_dist_{}.png'.format(name))

    for file_path in list_names:
        graph2 = read_graph2(file_path)
        name = file_path[-1]
        graph = dgl.to_networkx(graph2)
        graph = nx.Graph(graph)
        avg_short_path.append(calculate_avg_shortest_path(graph2))
        names.append(name)
        density.append(nx.density(graph))
        transitivity.append(nx.transitivity(graph))
        nodes.append(graph.number_of_nodes())
        edges.append(graph.number_of_edges())
        trans_dens.append(transitivity[-1]/density[-1])
        avg_degree.append(edges[-1]*2/nodes[-1])
        transitivity_density.append(transitivity[-1]>density[-1])
        degrees = list(dict(graph.degree()).values())
        var_degrees.append(np.var(degrees))
        avg_path_nodes.append(avg_short_path[-1]<np.log(nodes[-1]))
        skewed.append(skew(degrees))
        if draw == True:
            print("network name : "+names[-1])
            print()
            print('number of nodes : ',nodes[-1])
            print('nuler of edges : ', edges[-1])
            print('average_shortest_path', avg_short_path[-1])
            print("transitivity : ",transitivity[-1])
            print("density : ",density[-1])
            graph = nx.Graph(graph)

            degrees = [val for (node, val) in graph.degree()]
            plt.clf()
            plt.hist(degrees, bins=100)  # Adjust bins as needed
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.title("Degree Distribution")
            plt.show()
            plt.savefig('./degree_dist_{}.png'.format(name))

    data = {
    "Name": names,
    "Nodes": nodes,
    "Edges": edges,
    "Density": density,
    "Transitivity": transitivity,
    "Transitivity Density": trans_dens,
    "Average Shortest Path": avg_short_path,
    "Average Degree": avg_degree,
    'var Degree': var_degrees,
    "avg_path_nodes": avg_path_nodes,
    "transitivity_density": transitivity_density,
    "skew": skewed
    }
    df = pd.DataFrame(data)
    df.to_csv('metrics_of_stanford_networks.csv', index=False)  

    
    return names, density, transitivity, avg_short_path     



def download_Stanford_network(url, real_network_folder):
    
    input_file = real_network_folder+url.split("/")[-1]
    # subprocess.run(['wget', url, '-O', input_file])
    
    data = np.loadtxt(input_file, dtype=int)

    unique_values = np.unique(data)

    # map the IDs to a new range of values between 0 and k-1
    data = np.searchsorted(unique_values, data)

    data = data.tolist()
    data = [x for x in data]

    graph = dgl.graph(data)

    
    return graph, input_file

@torch.no_grad()
def test_network_diff_nfeat(model, graph, name, args):
    try:
        model.eval()
        ans = [] 
        softmax = torch.nn.Softmax(dim=1)
        #graph = dgl.add_self_loop(graph)
        graph = graph.to(args.device)
        # print(name + ' number of nodes is : ', graph.num_nodes())
        # print(name + ' number of edges is : ', graph.num_edges())
        
        if args.feat_type == 'ones_feat':
            graph.ndata['feat'] = torch.ones(graph.num_nodes(), 1).float().to(args.device)
        elif args.feat_type == 'noise_feat':
            graph.ndata['feat'] = torch.randn(graph.num_nodes(), 1).float().to(args.device)
        elif args.feat_type == 'identity_feat':
            graph.ndata['feat'] = compute_identity(torch.stack(graph.edges(), dim=0), graph.number_of_nodes(), args.k).float().to(args.device)
        elif args.feat_type == 'degree_feat':
            degrees = graph.in_degrees().unsqueeze(1).float().to(args.device)
            graph.ndata['feat'] = degrees.repeat(1, 1)
        else:
            degrees = graph.in_degrees().unsqueeze(1).float().to(args.device)
            graph.ndata['feat'] = degrees.repeat(1, 1)/(graph.number_of_nodes() - 1)

        logits = model(graph, args)
        
        real_name = name.split('/')[-1].split('.')[0]
        
        df = pd.read_csv('real_network.csv',)
        if real_name in df['network_name'].values:
            label = df[df['network_name'] == real_name][args.label_type].values.tolist()[0]
        else:
            label = None
        
        print("label : ",label)
        if args.data_type == "regression":
            result = logits
            #print([[result[0], abs(result[0]-label)]])
            ans.append([result[0].cpu().item(), label])
        else:
            result = softmax(logits)
            ans.append(result[0])
        #print('{} {} : '.format(name, args.feat_type), end='')
        if args.data_type == "regression":
            print('loss : {:.4f} '.format(result[0].item()), end='')
            print()
        else:
            for i, key in enumerate(['low', 'high']):
                print('{}. {:.4f} '.format(key, result[0][i]), end='')
            print()
    except Exception as e:
        print(f"An error occurred: {e}")
        ans = None
    print(ans[0])
    return ans

def radar_plot(ans, names, output_path, feat_type, param):
    ans = torch.tensor(ans)
    col =  list(param.keys())
    df = pd.DataFrame(ans.numpy()*100.0, index=names, columns=col)
    df.reset_index(inplace=True)

    # Plot data
    for i in range(len(df)):
        # number of variable
        categories=list(df)[1:]
        N = len(categories)
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Initialise the spider plot
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(projection='polar'),figsize=(15, 6))
        
        # Draw one axe per variable + add labels
        ax.set_xticks(angles[:-1], categories, color='grey', size=8)
        # Draw ylabele
        values=df.loc[i].drop('index').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.1)
        ax.set_rlabel_position(0)
        ax.set_yticks([25], ["25"], color="grey", size=7)
        ax.set_ylim(0,100)
        ax.set_title('{} model1'.format(feat_type))
        fig.suptitle(names[i])
        plt.show()
        plt.savefig('{}/radar{}_{}.pdf'.format(output_path, names[i], feat_type), format='pdf', bbox_inches='tight')

def test_networks2(model, args, param):
    G = nx.Graph()

    # Add 2000 nodes to the graph
    G.add_nodes_from(range(1, 2001))
    G = dgl.from_networkx(G)         
    value = test_network_diff_nfeat(model, G, 'graph1', args)
    print(value)
    G = nx.complete_graph(250)

    # Transfer the NetworkX graph to a DGL graph
    G = dgl.from_networkx(G)         
    value = test_network_diff_nfeat(model, G, 'graph2', args)
    print(value)
    
def test_networks3(model, args, param):
    seed = 0
    set_random_seed(seed)
    args.device = 'cpu'
    dataset = GraphDataset(device=args.device)
    dataset2 = GraphDataset(device=args.device)
    dataset.load('./data')
    dataset2.load('./test')
    generator = torch.Generator().manual_seed(seed)
    test_loader2 = GraphDataLoader(dataset2, batch_size=100, shuffle=True)
    num_training = int(len(dataset) * 0.9)
    num_val = int(len(dataset) * 0.)
    num_test = len(dataset) - num_val - num_training

    train_set, val_set, test_set = random_split(dataset, [num_training, num_val, num_test], generator=generator)
    #print(train_set.indices)
    # save test indices
    indices = torch.tensor(test_set.indices)
    ans_train = []
    ans_test = []
    for i in indices: 
        value = test_network_diff_nfeat(model, dataset.graphs[i], 'train', args)
        value.append(dataset.labels[i].value())
        ans_train.append(value)

    for i in range(len(dataset2)):
        value = test_network_diff_nfeat(model, dataset.graphs[i], 'test', args)
        value.append(dataset.labels[i].value())
        ans_test.append(value)
    
    print(ans_train)
    print(ans_test)      
    
def test_networks(model, args, result, data_folder):
    # Open the file in read mode
    ans = []
    names = []
    network_size = []
    with open('../data_folder/links.txt', 'r') as file:
        # Iterate over each line in the file
        f = 0
        for line in file:
            # Print the line
            if len(line) <= 1:
                f = 1
            else :
                if f == 1:
                    print()
                    print("field : "+line)
                    f = 0
                else:
                    graph, name = download_Stanford_network(line[:-1], data_folder)
                    network_size.append([graph.num_nodes(), graph.num_edges()])
                    value = test_network_diff_nfeat(model, graph, name, args)
                    if value != None:
                        ans.append(value[0])
                        names.append(name.split('/')[-1][:-7])
                    print()
            torch.cuda.empty_cache()

    for file_path in result:
        graph, name = read_graph(file_path)
        graph = dgl.from_networkx(graph)
        network_size.append([graph.num_nodes(), graph.num_edges()])
        value = test_network_diff_nfeat(model, graph, name, args)
        if value != None:
            ans.append(value[0])
            names.append(name)
        torch.cuda.empty_cache()

    for list_name in list_names:
        graph = read_graph2(list_name)
        name = list_name[-1]
        network_size.append([graph.num_nodes(), graph.num_edges()])
        value = test_network_diff_nfeat(model, graph, name, args)
        if value != None:
            ans.append(value[0])
            names.append(name)
        torch.cuda.empty_cache() 

    index = [(args.architecture, args.feat_type, name) for name in names]
    index = pd.MultiIndex.from_tuples(index, names=['model', 'feat_type', 'network_name'])
    list1, list2 = zip(*network_size)
    
    if args.data_type == "regression":
        df = pd.DataFrame(ans, columns=['pred_y', 'y'], index=index)
    else:
        df = pd.DataFrame(ans, columns=['Low', 'High'], index=index)
    df['num_nodes'] = list1
    df['num_edges'] = list2
    df.to_csv("{}/{}_stanford_output_testing2.csv".format(args.output_path, args.feat_type), index=True)
    print(df)
    #radar_plot(ans, names, args['output_path'], args['feat_type'], param)    

def extract_name_from_string(input_string):
    # Use case-insensitive regular expression to extract the last part of the path
    match = re.search(r'([^/]+)\.(net|gml)$', input_string, re.IGNORECASE)

    # Check if a match is found
    if match:
        name = match.group(1)
        return name
    else:
        return None  # Return None if no match is found

def download_and_extract(links, output_dir="../data_folder/extracted_folders"):
    os.makedirs(output_dir, exist_ok=True)

    net_files = []

    for link in links:
        if link.lower().endswith('.net'):
            # Download the .net file directly
            response = requests.get(link)
            net_file_path = os.path.join(output_dir, os.path.basename(link))
            with open(net_file_path, 'wb') as net_file:
                net_file.write(response.content)
            net_files.append(net_file_path)
        else:
            # Download the ZIP file
            response = requests.get(link)
            zip_file = zipfile.ZipFile(BytesIO(response.content))

            # Extract the ZIP file to the output directory
            folder_name = os.path.join(output_dir, os.path.splitext(os.path.basename(link))[0])
            zip_file.extractall(folder_name)

            # Find the first .net file in the extracted folder
            for root, dirs, files in os.walk(folder_name):
                for file in files:
                    if file.endswith(".net") or file.endswith('.gml'):
                        net_files.append(os.path.join(root, file))
                        break

    return net_files

def get_files_names(folder_path):
    return [folder_path+file for file in os.listdir(folder_path) if file.endswith(".gz.txt")]


def extract_numbers_from_string(input_string):
    # Use regular expression to find all numbers in the string
    numbers = list(map(int, re.findall(r'\d+', input_string)))

    return numbers

def create_graph(edges, num_nodes):
    # Create an empty graph
    graph = nx.Graph()

    # Add nodes to the graph
    graph.add_nodes_from(range(1, num_nodes + 1))

    # Add edges to the graph
    graph.add_edges_from(edges)

    return graph

def read_graph_gml_dataset(file_path):
    # Read the GML file into a NetworkX graph
    graph = nx.read_gml(file_path, destringizer=int, label='id')
    name = extract_name_from_string(file_path)
    # Remove all node attributes
    for node in graph.nodes():
        graph.nodes[node].clear()

    # Remove all edge attributes
    #for edge in graph.edges():
    #   graph.edges[edge].clear()

    return graph, name

def read_graph_pajek_datast(file_path):
    # Variables to store the graph information
    edges = []

    # Flag to indicate when to start reading edges
    reading_edges = False
    reading_arclist = False
    num_nodes = 0
    name = extract_name_from_string(file_path)
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.lower()

            if reading_edges:
                # Read edges in the format "x y"
                #x, y = map(int, line.split())
                numbers = extract_numbers_from_string(line)
                edges.append((numbers[0]-1, numbers[1]-1))
            elif reading_arclist:
                numbers = extract_numbers_from_string(line)
                for i in range(1, len(numbers)):
                    edges.append((numbers[0]-1, numbers[i]-1))
            elif line.startswith("*edges"):
                # Start reading edges
                reading_edges = True
            elif line.startswith("*arcslist"):
                reading_arclist = True
            elif line.startswith("*vertices"):
                numbers = extract_numbers_from_string(line)
                num_nodes = numbers[0]-1

    # Create a graph using NetworkX
    graph = create_graph(edges, num_nodes)

    # You can now work with the 'graph' object using NetworkX functions
    return graph, name

def read_graph(file_path):
    if file_path.endswith('.net'):
        graph, name = read_graph_pajek_datast(file_path)
    elif file_path.endswith('.gml'):
        graph, name = read_graph_gml_dataset(file_path)

    return graph, name

def graph_statistics(result, draw = False):
    # Open the file in read mode
    names = []
    nodes = []
    edges = []
    density = []
    transitivity = []
    trans_dens = []
    avg_short_path = []
    avg_degree = []

    for file_path in result:
        graph, name = read_graph(file_path)
        grpah2 = dgl.from_networkx(graph)
        avg_short_path.append(calculate_avg_shortest_path(grpah2))
        names.append(name)
        density.append(nx.density(graph))
        transitivity.append(nx.transitivity(graph))
        nodes.append(graph.number_of_nodes())
        edges.append(graph.number_of_edges())
        trans_dens.append(transitivity[-1]/density[-1])
        avg_degree.append(edges[-1]*2/nodes[-1])
        if draw == True:
            print("network name : "+names[-1])
            print()
            print('number of nodes : ',nodes[-1])
            print('nuler of edges : ', edges[-1])
            print('average_shortest_path', avg_short_path[-1])
            print("transitivity : ",transitivity[-1])
            print("density : ",density[-1])
            graph = nx.Graph(graph)

            degrees = [val for (node, val) in graph.degree()]
            plt.hist(degrees, bins=100)  # Adjust bins as needed
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.title("Degree Distribution")
            plt.show()

    for file_path in list_names:
        graph2 = read_graph2(file_path)
        name = file_path[-1]
        graph = dgl.to_networkx(graph2)
        graph = nx.Graph(graph)
        avg_short_path.append(calculate_avg_shortest_path(graph2))
        names.append(name)
        density.append(nx.density(graph))
        transitivity.append(nx.transitivity(graph))
        nodes.append(graph.number_of_nodes())
        edges.append(graph.number_of_edges())
        trans_dens.append(transitivity[-1]/density[-1])
        avg_degree.append(edges[-1]*2/nodes[-1])
        if draw == True:
            print("network name : "+names[-1])
            print()
            print('number of nodes : ',nodes[-1])
            print('nuler of edges : ', edges[-1])
            print('average_shortest_path', avg_short_path[-1])
            print("transitivity : ",transitivity[-1])
            print("density : ",density[-1])
            graph = nx.Graph(graph)

            degrees = [val for (node, val) in graph.degree()]
            plt.hist(degrees, bins=100)  # Adjust bins as needed
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.title("Degree Distribution")
            plt.show()

    data = {
    "Name": names,
    "Nodes": nodes,
    "Edges": edges,
    "Density": density,
    "Transitivity": transitivity,
    "Transitivity Density": trans_dens,
    "Average Shortest Path": avg_short_path,
    "Average Degree": avg_degree
    }
    df = pd.DataFrame(data)
    df.to_csv('metrics_of_stanford_networks.csv', index=False)


    return names, density, transitivity, avg_short_path

def fill_arg_parser(parser, args_dict):
    for key, value in args_dict.items():
        if hasattr(parser, key.lstrip('-')):
            setattr(parser, key.lstrip('-'), value)
    return parser

if __name__ == "__main__":
    
    args2 = parse_args2()

    if args2.dist_draw == True:
        result = download_and_extract(linkss)
        stanford_degree_dist_plots(result)
        #graph_statistics(result, True)
    else:
        with open(args2.args_file, 'r') as f:
            args = json.load(f)
        args_dict = args['params']

        args = parse_args()
        for key, value in args_dict.items():
            setattr(args, key, value)
        args.output_path = args2.output_path
        model_op = get_network(args.architecture)
        model = model_op(
            in_dim=args.num_feature,
            hidden_dim=args.hidden_dim,
            out_dim=args.num_classes,
            num_layers=args.num_layers,
            pool_ratio=args.pool_ratio,
            dropout=args.dropout,
            output_activation = args.output_activation
        ).to(args.device)
        model.load_state_dict(torch.load(args2.model_weights_path))
        #args['feat_type'] = args2.feat_type
        result = download_and_extract(linkss)
        #result = get_files_names(args2.link_data)
        test_networks(model, args, result, args2.real_network_folder)
        #test_networks3(model, args, param)
        



@torch.no_grad()
def test_network_diff_nfeat_simple(model, graph, name):
    label_type = "density"
    feat_type="ones_feat"
    device = "cuda"
    k = 1
    try:
        model.eval()
        ans = [] 
        #graph = dgl.add_self_loop(graph)
        graph = graph.to(device)
        # print(name + ' number of nodes is : ', graph.num_nodes())
        # print(name + ' number of edges is : ', graph.num_edges())
        
        if feat_type == 'ones_feat':
            graph.ndata['feat'] = torch.ones(graph.num_nodes(), 1).float().to(device)
        elif feat_type == 'noise_feat':
            graph.ndata['feat'] = torch.randn(graph.num_nodes(), 1).float().to(device)
        elif feat_type == 'identity_feat':
            graph.ndata['feat'] = compute_identity(torch.stack(graph.edges(), dim=0), graph.number_of_nodes(), k).float().to(device)
        elif feat_type == 'degree_feat':
            degrees = graph.in_degrees().unsqueeze(1).float().to(device)
            graph.ndata['feat'] = degrees.repeat(1, 1)
        else:
            degrees = graph.in_degrees().unsqueeze(1).float().to(device)
            graph.ndata['feat'] = degrees.repeat(1, 1)/(graph.number_of_nodes() - 1)

        logits = model(graph)
        #print("logits: ",logits)
        real_name = name.split('/')[-1].split('.')[0]
        label = graph.in_degrees().unsqueeze(1).float().mean().item()

        # df = pd.read_csv('real_network.csv',)
        # if real_name in df['network_name'].values:
        #     label = df[df['network_name'] == real_name][label_type].values.tolist()[0]
        # else:
        #     print('error')
        #     label = None
        
        #print("label : ",label)
        
        result = logits
        #print(result[0][0])
        #print([[result[0], abs(result[0]-label)]])
        ans = [result[0].cpu().item(), label]
        
        # print('loss : {:.4f} '.format(result[0].item()), end='')
        # print()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        ans = None
    loss = (result[0][0].cpu().item()-label)**2
    
    return loss


def test_networks_simple(model, result, data_folder):
    # Open the file in read mode
    ans = []
    names = []
    network_size = []
    with open('../data_folder/links.txt', 'r') as file:
        # Iterate over each line in the file
        f = 0
        for line in file:
            # Print the line
            if len(line) <= 1:
                f = 1
            else :
                if f == 1:
                    # print()
                    # print("field : "+line)
                    f = 0
                else:
                    graph, name = download_Stanford_network(line[:-1], data_folder)
                    network_size.append([graph.num_nodes(), graph.num_edges()])
                    value = test_network_diff_nfeat_simple(model, graph, name)
                    if value != None:
                        ans.append(value)
                        names.append(name.split('/')[-1][:-7])
                    # print()
            torch.cuda.empty_cache()

    for file_path in result:
        graph, name = read_graph(file_path)
        graph = dgl.from_networkx(graph)
        network_size.append([graph.num_nodes(), graph.num_edges()])
        value = test_network_diff_nfeat_simple(model, graph, name)
        if value != None:
            ans.append(value)
            names.append(name)
        torch.cuda.empty_cache()

    for list_name in list_names:
        graph = read_graph2(list_name)
        name = list_name[-1]
        network_size.append([graph.num_nodes(), graph.num_edges()])
        value = test_network_diff_nfeat_simple(model, graph, name)
        if value != None:
            ans.append(value)
            names.append(name)
        torch.cuda.empty_cache() 
    architecture = "simple_gnn"
    feat_type = "ones_feat"
    index = [(architecture, feat_type, name) for name in names]
    index = pd.MultiIndex.from_tuples(index, names=['model', 'feat_type', 'network_name'])
    list1, list2 = zip(*network_size)
    
    print("loss of real network : ",sum(ans) / len(ans))
    return sum(ans) / len(ans)

        
def run_real_networks(model):
    result = download_and_extract(linkss)
    real_network_folder = "../data_folder/"
    loss = test_networks_simple(model, result, real_network_folder)
    return loss