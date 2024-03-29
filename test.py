import torch
import torch.nn as nn

# Define the linear layer
m = nn.Linear(20, 30)

# Generate random input tensor
input = torch.randn(128, 20)

# Pass the input through the linear layer
output1 = m(input)
output2 = m(input)

print((output1 == output2).all())

# Print the size of the output tensor
print(output.size())
