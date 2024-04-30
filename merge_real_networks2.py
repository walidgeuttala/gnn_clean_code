import pandas as pd

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
    dfs1 = []
    names1 = []
    for idx1, feat in enumerate(feats):
        dfs2 = []
        names2 = []
        for idx2, model in enumerate(models):
            idx = (idx1)*len(models)+idx2+1
            path = f"../gnn_outputs/{folder}/output{idx}/{feat}_stanford_output_testing.csv"
            df = pd.read_csv(path)
            names2.append(model)
            dfs2.append(df.iloc[:, 2:])
        dfs1.append(pd.concat(dfs2, keys=names2))
        names1.append(feat)
    dfs.append(pd.concat(dfs1, keys=names1))
    names.append(folder)

merged_df = pd.concat(dfs, keys=names)
merged_df.to_csv('all_real_world_networks_results.csv')
print(merged_df)

print(merged_df.loc['density'].groupby(['network_name']).agg({'Loss': ['min', 'mean', 'max']}))