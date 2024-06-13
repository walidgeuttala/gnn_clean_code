import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.dataloading import GraphDataLoader

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)
        return out

# Assuming node features have 1 dimension
input_dim = 1
hidden_dim = 128
num_layers = 2
output_dim = 1  # Predicting a single value: average degree

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# Example data preparation (using random data for demonstration purposes)
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    node_features = batched_graph.ndata['feat']
    node_features = node_features.unsqueeze(-1)  # Adding a dimension to match input_dim
    labels = torch.tensor(labels, dtype=torch.float32)
    return node_features, labels

# Dummy dataset and dataloader
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

# Example usage:
graphs = [dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 0]))), 
          dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])))]
for g in graphs:
    g.ndata['feat'] = torch.randn(g.number_of_nodes(), 1)

labels = [1.0, 2.0]  # Example labels for average degree

dataset = GraphDataset(graphs, labels)
dataloader = GraphDataLoader(dataset, collate_fn=collate, batch_size=2)

# Training loop (simplified)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for node_features, labels in dataloader:
        outputs = model(node_features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Note: This is a simplified example. You would need to adapt it to your specific use case, 
# such as handling graphs of different sizes and possibly normalizing the features.
