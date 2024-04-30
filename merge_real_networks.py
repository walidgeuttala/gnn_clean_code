import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

feats = ['ones_feat', 'noise_feat', 'degree_feat', 'norm_degree_feat', 'identity_feat']

models = ['gin', 'gat', 'global', 'hierarchical', 'gatv2']
feats = ['identity_feat']
models = ['gin']
main_folders = ['density', 'transitivity' , 'kurtosis', 'average_path']
cnt = 1
number_folders = 1
dfs = []
names = []

for folder in main_folders:
    for idx in range(number_folders):
        path = f"../gnn_outputs/{folder}/output{idx+1}/{feats[idx//len(models)]}_stanford_output_testing2.csv"
        df = pd.read_csv(path)
        df['Loss'] = abs(df['y']-df['pred_y'])
        names.append(f'{folder}')
        dfs.append(df)
merged_df = pd.concat(dfs, keys=names)
merged_df.to_csv('all_real_world_networks_results.csv')
subnames = ['facebook_combined', 'wiki-Vote', 'p2p-Gnutella04', 'p2p-Gnutella08', 'CSphd', 'geom', 'adjnoun', 'football', 'hep-th', 'netsience', 'CLUSTERDataset', 'TreeGridDataset']
namess = ["MUTAG", "COLLAB", "ENZYMES"]
values = ['pred_y', 'y', 'Loss']
for value in values:
    for name in main_folders:
        df = merged_df.loc[name]
        df = df[df['network_name'].isin(subnames)]
        print(df)
        # Create two boxplots using seaborn
        plt.figure(figsize=(10, 6))

        # First boxplot for num_nodes
        plt.subplot(1, 2, 1)
        sns.boxplot(x='num_nodes', y=value, data=df)
        plt.title('Boxplot for num_nodes')
        plt.xticks(rotation=45)
        # Second boxplot for num_edges
        plt.subplot(1, 2, 2)
        sns.boxplot(x='num_edges', y=value, data=df)
        plt.title('Boxplot for num_edges')
        plt.suptitle(f"{name}_{namess[0]}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plot_{name}_{namess[1]}_{value}.png')

# for feat in feats:
#     for i in range(5):
#         df = pd.read_csv(f"output{cnt}/"+feat+"_stanford_output_testing.csv")
#         dfs.append(df)
#         names.append(models[i]+'_'+feat)
#         cnt += 1 

# merged_df = pd.concat(dfs, keys=names)
# merged_df.to_csv('real_world_networks_results.csv')
# print(merged_df)
