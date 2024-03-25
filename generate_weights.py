import torch
import os
from utils import set_random_seed
from network import get_network
from itertools import product

# Directory to save weights
save_dir = "../weights"
os.makedirs(save_dir, exist_ok=True)

# Define search space
search_space = {
    "feat_type": ['ones_feat', 'noise_feat', 'degree_feat', 'norm_degree_feat', 'identity_feat'],
    "architecture": ["hierarchical", "global", "gin", "gatv2"],
    "hidden_dim": [4, 8, 16, 32, 64],
    "lr": [1e-2],
    "num_layers": [3, 4],
    "weight_decay": [1e-3],
    "k": [4],
    "dropout": [0.],
    "pool_ratio": [0.9],
    "output_activation": ['Identity'],
    "data_type": ['regression'],
}

# Perform 5 trials
for trial in range(5):
    print(f"Trial {trial + 1}:")
    # Set random seed for this trial
    set_random_seed(trial)
    
    # Iterate over each combination in the search space
    for params in product(*search_space.values()):
        feat_type, architecture, hidden_dim, lr, num_layers, weight_decay, k, dropout, pool_ratio, output_activation, data_type = params
        
        # Determine in_dim based on feat_type
        if feat_type == 'identity_feat':
            in_dim = k
        else:
            in_dim = 1

        # Create the network based on the current configuration
        model = get_network(architecture)(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,  # Replace with appropriate output dimension
            num_layers=num_layers,
            pool_ratio=pool_ratio,
            dropout=dropout,
            output_activation=output_activation
        )

        # Save weights
        save_path = os.path.join(save_dir, f"trial_{trial+1}_{feat_type}_{architecture}_{hidden_dim}_{num_layers}_{lr}_{weight_decay}_{k}_{dropout}_{pool_ratio}_{output_activation}_{data_type}_weights.pth")
        torch.save(model.state_dict(), save_path)
