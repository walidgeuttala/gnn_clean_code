import os 
import json
import pandas as pd
import numpy as np
import re
def count_output_folders():
    current_directory = os.getcwd()
    all_items = os.listdir(current_directory)
    output_folders = [folder for folder in all_items if os.path.isdir(folder) and folder.startswith('output')]
    return len(output_folders)

def extract_float_values(input_string):
    # Define a regular expression pattern to match the two float values
    pattern = r'([\d.]+)\(\+-([\d.]+)\)'

    # Use re.match to find the matches in the string
    match = re.match(pattern, input_string)

    if match:
        # Extract the float values from the matched groups
        value1 = float(match.group(1))
        value2 = float(match.group(2))

        return [value1, value2]
    else:
        print("Pattern not found in the string.")
        return None

length = 39
current_path = '../gnn_outputs/transitivity/'
#keys = ['architecture', 'hidden_dim', 'num_layers', 'feat_type', 'train_loss', 'train_loss_error', 'train_acc', 'train_acc_error', 'valid_acc', 'valid_acc_error', 'test_acc', 'test_acc_error']
keys = ['architecture', 'hidden_dim', 'num_layers', 'feat_type', 'test_acc', 'test_acc2']
df = []
for i in range(1, length, 2):
    args_path = [f for f in os.listdir(current_path+'output{}/'.format(i)) if f.startswith('Data_dataset')][0]
    with open(current_path+'output{}/'.format(i)+args_path) as f:
        data = json.load(f)
    #print(data['result'][0])
    #results = np.array(data['results'])
    #ans = np.ones(results.shape[1]*2)
    results = extract_float_values(data['result'])
    results = data['results']
    mean_results1 = np.mean(np.array(results[0]))
    mean_results2 = np.mean(np.array(results[1]))
    #var_results = np.var(results, axis=0)
    #ans[::2] = mean_results
    #ans[1::2] = var_results
    values = [data['hyper-parameters']['architecture'], data['hyper-parameters']['hidden_dim'], data['hyper-parameters']['num_layers'], data['hyper-parameters']['feat_type']]+[mean_results1, mean_results2]
    
    my_dict = {keys[i]: values[i] for i in range(len(keys))}
    df.append(my_dict)

df = pd.DataFrame(df)
df.to_csv('test_resutls_small_networks.csv')

print(df)
