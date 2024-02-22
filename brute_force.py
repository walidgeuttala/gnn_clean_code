import torch
import subprocess
import json 
import os 
import re

#feat_types = ['ones_feat', 'noise_feat', 'degree_feat', 'norm_degree_feat', 'identity_feat']
feat_types = ['ones_feat']
save_last_epoch_hidden_output = False

num_trials = 1
epochs = 10
epoch_search = 10
cnt = 0
device = 'cuda'
label_type = 'density'
data_type = 'classification'
loss_name = 'NLLLoss' # MSELoss
output_activation = 'LogSoftmax' # Identity

def find(str2, dir_path):
  files = os.listdir(dir_path)
  file_names = [os.path.basename(file) for file in files]

  matching_names = list(filter(lambda x: re.search(str2, x, re.IGNORECASE), file_names))

  # Pass the folder name to the command line
  return matching_names[0]

#models = ['gin', 'gat', 'global', 'hierarchical', 'gatv2']
models = ['gin']

search_space = {
    "architecture": ['gin'],
    "hidden_dim": [16],
    "lr": [1e-2],
    "num_layers":[3],
    "weight_decay": [1e-3],
    "k": [4]
}

for j, feat_type in enumerate(feat_types):
    print()
    print('{}-------------------------------feat_type : {}---------------------'.format(j+2 ,feat_type))
    print()
    for k, model in enumerate(models):
      cnt += 1
      search_space['architecture'] = [model]
      print()
      print('{}.{}--------------------------------------------model name  : {}---------------------'.format(j+2,k+1,search_space['architecture']))
      print()

      with open('grid_search_config.json', 'w') as fp:
        json.dump(search_space, fp)

      print()
      print('{}.{}.1--------------------------------------------Grid_Search  : {}---------------------'.format(j+2,k+1,search_space['architecture']))
      print()
      output_path = f'../gnn_outputs/version1/output{cnt}/'
      # os.makedirs(output_path, exist_ok=True)
  
      script = f"python grid_search.py --feat_type {feat_type} --epochs {epoch_search} --output_path {output_path} --device {device} --label_type {label_type} --data_type {data_type} --loss_name {loss_name} --output_activation {output_activation}"
      script = script.split()
      subprocess.run(script)

      torch.cuda.empty_cache()

      # str1 = output_path+find('model', output_path)
      # print(str1)
      with open(output_path+"model.log", 'r') as f:
          modell = json.load(f)

      model = modell['params']
      model['epochs'] = epochs
      model['save_last_epoch_hidden_output'] = save_last_epoch_hidden_output
      model['patience'] = -1
      model['num_trials'] = num_trials 
      formatted_string = " ".join([f"--{key} {value}" for key, value in model.items()])

      print()
      print('{}.{}.2--------------------------------------------Train and Test  : {}---------------------'.format(j+2,k+1,search_space['architecture']))
      print()
      script = "python main.py " + formatted_string
      script = script.split()
      subprocess.run(script)

      torch.cuda.empty_cache()

      model = modell['params2']
      model['epochs'] = epochs
      model['save_last_epoch_hidden_output'] = save_last_epoch_hidden_output
      model['patience'] = -1
      model['num_trials'] = num_trials 
      model['changer'] = 1
      formatted_string = " ".join([f"--{key} {value}" for key, value in model.items()])

      #print()
      #print('{}.{}.2--------------------------------------------Train and Test  : {}---------------------'.format(j+2,k+1,search_space['architecture']))
      #print()
      #script = "python main.py " + formatted_string
      #script = script.split()
      #subprocess.run(script)

      torch.cuda.empty_cache()

      # plots analysis of hidden features
      #for i in range(3):
      #  comparing_hidden_feat('data', 'output', 250, i)

      #str1 = 'output/'+find('last')
      #str2 = 'output/'+find('Data_')
      #script = "python test_stanford_networks.py --model_weights_path {} --args_file {} --dataset_path {} --feat_type {}".format(str1, str2, data_path, model['feat_type'])
      #script = script.split()
      #subprocess.run(script)
      
      cnt += 1