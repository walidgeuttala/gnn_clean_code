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

import torch
from torch.utils.data import Dataset

class RandomIntDataset(Dataset):
    def __init__(self, n_min, n_max, k, num_samples=5000):
        self.n_min = n_min
        self.n_max = n_max
        self.k = k
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random sequence length between n_min and n_max
            n = 100
            
            # Generate sequence with k features
            sample = torch.randint(0, 101, (n, k), dtype=torch.float)
            
            # Append the sample to data
            self.data.append(sample)
            
            # Use the average of the first column as the label
            label = sample[:, 0].mean().item()
            self.labels.append(label)

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

import inspect

def debug(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_name = [name for name, val in callers_local_vars if val is var]
    if var_name:
        print(f"{var_name[0]}: {var}")
    else:
        print(f"Variable name could not be determined. Value: {var}")


def get_dataloaders(n, feat, batch_size=32, num_samples=5000):
    dataset = RandomIntDataset(10, n, feat, num_samples)
    
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # collate_fn=collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims=[32], output_dim=1):
        super(MLP, self).__init__()
        
        # Creating the list of layers
        layers = []
        in_dim = input_size
        
        # Adding hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            #layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        # Adding the final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        # Converting list to a ModuleList
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, nhead=8, dim_feedforward=256, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Apply embedding
        x = self.embedding(x)
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Transpose for transformer: (seq_len, batch_size, hidden_size)
        x = x.transpose(0, 1)
        
        # Forward propagate through the transformer encoder
        x = self.transformer_encoder(x)
        
        # We are using the output from the last time step
        x = x[-1, :, :]
        
        # Pass through the fully connected layer
        out = self.fc(x)
        
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Weight initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We are using the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out


import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
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
        scheduler.step()
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
    print('simple_code_mlp code')
    n = 100  # Maximum number of random integers in each sample
    num_samples = 5000
    batch_size = 32
    num_epochs = 100
    trials = 1
    train_losses = 0
    valid_losses = 0
    test_losses  = 0
    feat = 1
    for i in range(trials):
        set_random_seed(i)
        train_loader, val_loader, test_loader = get_dataloaders(n, feat, batch_size, num_samples)
        
        model = TransformerModel(input_size=feat)
        
        train_loss, valid_loss = train_model(model, train_loader, val_loader, num_epochs)
        print("train_loss : ", train_loss)
        train_losses += train_loss
        valid_losses += valid_loss
        test_losses += evaluate_model(model, test_loader)
    print(f"Train loss average: {train_losses/trials:.6f}, Valid loss average: {valid_losses/trials:.6f}, Test loss average: {test_losses/trials:.6f}")

if __name__ == "__main__":
    main()
