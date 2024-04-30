import pandas as pd
import torch 
import numpy as np

properties = ['average_path', "transitivity", 'kurtosis', 'density']
data_names = ['MUTAG', 'ENZYMES', 'DD', 'COLLAB']
# self.data_types = ['classification', 'regression']

y = torch.load(f"../data_folder/dgl_graph_labels/{data_names[0]}_properties_labels.pt")[8]
print(y)
df = pd.read_csv('real_network.csv', index_col=0)
df1 = df[[properties[3]]]
print(df)
print(df1)

df2 = pd.DataFrame(y)
print(df2)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df1' is your first DataFrame with multiple instances and one column
# Assuming 'df2' is your second DataFrame with one column

# Create a histogram for df1
plt.figure(figsize=(8, 6))
sns.histplot(data=df1, x=df1.columns[0], bins=20, kde=True)

# Overlay the points from df2 onto the histogram
plt.plot(df2[df2.columns[0]], [0]*len(df2), 'ro', markersize=5)  # 'ro' for red circles

# Set labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram with Overlayed Points')

# Save the plot
plt.savefig('dist.png')

# Show plot
plt.show()

