import pandas as pd
import numpy as np
import os

data = [[ 6.41696155e-01,  5.19174278e-01,  3.40257568e+00,  4.36910126e+01],
        [-2.67103091e-02,  1.25479149e-01,  2.28318661e+02,  2.83238229e+01],
        [ 2.44098353e+00,  5.40202894e-03,  1.79801635e+01,  7.35454211e+00],
        [ 1.93902540e+00,  2.06598793e-02,  2.77405551e+01,  6.59482622e+00],
        [ 2.79581213e+00,  3.62264151e-03,  1.39843053e+01,  1.84909671e+00],
        [ 5.35896480e-01,  2.40429585e-01,  1.68792206e+01,  3.24063734e+00],
        [-5.88242054e-01,  6.93441415e-01,  1.42862943e+00,  3.45122719e+00],
        [ 2.53555346e+00,  1.56934979e-01, -1.29309994e+00,  7.58928571e+00],
        [ 2.50816178e+00,  4.07239819e-01,  1.15008386e-01,  1.06608696e+01],
        [ 2.90918350e+00,  3.29575580e-01,  6.03926905e+00,  3.76773113e+00],
        [ 1.69761276e+00,  3.15569196e-01, -8.45168551e-01,  3.50769231e+01],
        [-9.64333296e-01,  0.00000000e+00, -4.35806002e-01,  2.77010561e+00]]

# Creating DataFrame
subnames = ['facebook_combined', 'wiki-Vote', 'p2p-Gnutella04', 'p2p-Gnutella08', 'CSphd', 'geom', 'adjnoun', 'football', 'hep-th', 'netsience', 'CLUSTERDataset', 'TreeGridDataset']
df2 = pd.DataFrame(data, columns=["average_path",  'transitivity', 'kurtosis', 'density'], index=subnames)


number_folder = 4

names = []
min_values = []
max_values = []
mean_values = []
column = 'kurtosis'
folder_path = f'../gnn_outputs/{column}/output'
for i in range(number_folder):
    folder_path_n = folder_path+f"{i+1}/"
    listt = [file for file in os.listdir(folder_path_n) if file.endswith(".csv")]
    if len(listt) == 0:
        continue
    file = folder_path_n+[file for file in os.listdir(folder_path_n) if file.endswith(".csv")][0]
    df = pd.read_csv(file)
    if df.shape[0] == 0 and df.shape[1] == 0:
        continue
    
    df.set_index('network_name', inplace=True)
    merged_df = pd.merge(df, df2, left_index=True, right_index=True, how='inner')
    merged_df['loss'] = (merged_df['Loss'] - merged_df[column]).abs()
    if i == 4:
        print(merged_df)
    names.append(df.iloc[0, 0]+" "+df.iloc[0, 1])
    min_val = merged_df['loss'].min()
    max_val = merged_df['loss'].max()
    mean_val = merged_df['loss'].mean()
    min_values.append(min_val)
    max_values.append(max_val)
    mean_values.append(mean_val)
    

statistics_df = pd.DataFrame({
    "Name": names,
    'Min': min_values,
    'Mean': mean_values,
    'Max': max_values
})

print(statistics_df)
