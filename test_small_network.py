import networkx as nx
import dgl
import torch
from data import GraphDataset
from network import *
from utils import *
from dgl.dataloading import GraphDataLoader

graphs = []
n = 8
degree = 4
device = 'cpu'

@torch.no_grad()
def test_regression(model: torch.nn.Module, loader):
    model.eval()
    loss = 0.0
    num_graphs = 0
    loss_func = getattr(F, 'MSELoss')(reduction="sum")
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += 1
        out = model(batch_graphs, args)
        loss += loss_func(out, batch_labels).item()

    return loss / num_graphs

def main(args, seed, save=True):
    set_random_seed(seed)
    # ER_low
    graphs.append(nx.gnp_random_graph(n, degree / n, seed=1, directed=False))
    labels = torch.tensor([sum(dict(nx.degree(graphs[0])).values()) / len(graphs[0])])
    for G in graphs:
        degrees = dict(G.degree())
        nx.set_node_attributes(G, degrees, 'feat')

    graphs = [dgl.from_networkx(graph) for graph in graphs]
    dataset = GraphDataset(graphs, labels, device)

    test_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    
    
    # Step 2: Create model =================================================================== #
    set_random_seed(seed)
    model = SAGNetworkHierarchical(
            in_dim = 1,
            hidden_dim = 2,
            out_dim = 1,
            num_layers=2,
            pool_ratio = 0.2,
            dropout = 0.0,
            output_activation = 'Identity'
        )
    
    # Step 3: Create training components ===================================================== #
    optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=0.2, weight_decay=0.0)  # Replace `parameters` with your specific parameters


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    
    
    loss_value = test_regression(model, test_loader)
        
    return loss_value
