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
    def __init__(self, n_min, n_max, k, num_samples=5000):
        self.n_min = n_min
        self.n_max = n_max
        self.k = k
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random sequence length between n_min and n_max
            n = random.randint(n_min, n_max)
            
            # Generate sequence with k features
            sample = torch.randint(0, 101, (n, k), dtype=torch.float32)
            
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
    lengths = torch.tensor([len(d) for d in data])
    return data_padded, labels, lengths

def get_dataloaders(n, feat, batch_size=32, num_samples=5000):
    dataset = RandomIntDataset(10, n, feat, num_samples)
    
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.):
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

    def forward(self, x, lengths):
        # Pack the padded batch of sequences
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out_packed, _ = self.lstm(x_packed, (h0, c0))
        
        # Unpack the output
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        
        # Use the output from the last non-padded element in each sequence
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(1)
        out = out.gather(1, idx).squeeze(1)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out
    
from genagg.genagg import GenAgg

class GenAggModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.):
        super(GenAggModel, self).__init__()
        # Replace LSTM with GenAgg
        self.aggregation = GenAgg()
        self.fc = nn.Linear(input_size, 1)  # Assuming you're aggregating and reducing over input features
        self.dropout = nn.Dropout(dropout)

        # Initialize the fully connected layer weights
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(self, x, lengths):
        # Use GenAgg for aggregation
        # No need for packing sequences as we're directly aggregating over the input
        x_aggregated = self.aggregation(x)
        
        # Apply dropout
        out = self.dropout(x_aggregated)
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out


def train_model(model, train_loader, val_loader, num_epochs=20):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels, lengths in train_loader:
            inputs, labels = inputs.float(), labels.float()  # Ensure data is float32
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels = inputs.float(), labels.float()  # Ensure data is float32
                outputs = model(inputs, lengths)
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
        for inputs, labels, lengths in test_loader:
            inputs, labels = inputs.float(), labels.float()  # Ensure data is float32
            outputs = model(inputs, lengths)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item() * inputs.size(0)
    
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    return test_loss

def main():
    n = 10  # Maximum number of random integers in each sample
    num_samples = 10000
    batch_size = 64
    num_epochs = 1000
    trials = 10
    train_losses = 0
    valid_losses = 0
    test_losses = 0
    feat = 1
    
    for i in range(trials):
        set_random_seed(i)
        train_loader, val_loader, test_loader = get_dataloaders(n, feat, batch_size, num_samples)
        print("GenAggModel")
        model = GenAggModel(input_size=feat)
        
        train_loss, valid_loss = train_model(model, train_loader, val_loader, num_epochs)
        print("train_loss : ", train_loss)
        train_losses += train_loss
        valid_losses += valid_loss
        test_losses += evaluate_model(model, test_loader)
    
    print(f"Train loss average: {train_losses/trials:.6f}, Valid loss average: {valid_losses/trials:.6f}, Test loss average: {test_losses/trials:.6f}")

if __name__ == "__main__":
    main()
