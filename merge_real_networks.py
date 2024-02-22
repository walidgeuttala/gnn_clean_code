import pandas as pd

feats = ['ones_feat', 'noise_feat', 'degree_feat', 'norm_degree_feat', 'identity_feat']

models = ['gin', 'gat', 'global', 'hierarchical', 'gatv2']
cnt = 1
dfs = []
names = []
for feat in feats:
    for i in range(5):
        df = pd.read_csv(f"output{cnt}/"+feat+"_stanford_output_testing.csv")
        dfs.append(df)
        names.append(models[i]+'_'+feat)
        cnt += 1 

merged_df = pd.concat(dfs, keys=names)
merged_df.to_csv('real_world_networks_results.csv')
print(merged_df)
