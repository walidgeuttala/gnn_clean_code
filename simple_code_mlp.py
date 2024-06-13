import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random 
import numpy as np
import dgl

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

class RandomIntDataset(Dataset):
    def __init__(self, n_min, n_max, num_samples=5000):
        self.n_min = n_min
        self.n_max = n_max
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            n = torch.randint(n_min, n_max + 1, (1,)).item()
            sample = torch.randint(0, 101, (n, 1), dtype=torch.float)  # Add extra dimension for LSTM input
            self.data.append(sample)
            self.labels.append(sample.mean())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch):
    data, labels = zip(*batch)
    data = [torch.tensor(d, dtype=torch.float32) for d in data]
    labels = torch.tensor(labels, dtype=torch.float32)
    data_padded = torch.nn.utils.rnn.pad_sequence(data, batch_first=True)
    return data_padded, labels

def get_dataloaders(n, batch_size=32, num_samples=5000):
    dataset = RandomIntDataset(10, n, num_samples)
    
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We are using the output from the last time step
        out = out[:, -1, :]
        
        # Pass through the fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_loss, val_loss

def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item() * inputs.size(0)
    
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss

def main():
    n = 100  # Maximum number of random integers in each sample
    num_samples = 5000
    batch_size = 32
    num_epochs = 100
    trials = 10
    train_losses = 0
    valid_losses = 0
    test_losses  = 0
    for i in range(trials):
        set_random_seed(i)
        train_loader, val_loader, test_loader = get_dataloaders(n, batch_size, num_samples)
        
        model = LSTMModel(input_size=1)
        
        train_loss, valid_loss = train_model(model, train_loader, val_loader, num_epochs)
        train_losses += train_loss
        valid_losses += valid_loss
        test_losses += evaluate_model(model, test_loader)
    print(f"Train loss average: {train_losses/trials}, Valid loss average: {valid_losses/trials}, Test loss average: {test_losses/trials}")

if __name__ == "__main__":
    main()
