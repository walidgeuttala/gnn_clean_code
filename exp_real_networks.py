import shutil
import os
import json
import re 
import subprocess

def find(str2, dir_path):
  # Set the directory path

  files = os.listdir(dir_path)
  file_names = [os.path.basename(file) for file in files]

  matching_names = list(filter(lambda x: re.search(str2, x, re.IGNORECASE), file_names))

  # Pass the folder name to the command line
  return matching_names[0]

def find_highest_accuracy_index(combined_lists):

    # Sort the list of tuples based on the accuracy in the second list (test2) and then the first list (test1) in descending order
    sorted_list = sorted(combined_lists, key=lambda x: (x[1], x[0]), reverse=True)

    # Retrieve the index of the highest accuracy model from the sorted list
    highest_accuracy_index = combined_lists.index(sorted_list[0])

    return highest_accuracy_index

number_folders = 25
data_path = 'data'
#for i in range(18, number_folders+1): 
for i in range(1, 25+1): 
    current_folder_name = 'output{}'.format(i)
    os.rename(current_folder_name, 'output')
    files_names = os.listdir('output')
    #str1 = 'output/'+find('last', 'output/')
    str3 = 'output/'+[file for file in files_names if file.startswith("Data_dataset_Hidden")][0]
    print(str3)
    with open(str3) as f:
        data = json.load(f)
    index = find_highest_accuracy_index(data['results'])
    #index = 0
    str1 = 'output/'+[file for file in files_names if file.startswith(f"last_model_weights_trail{index}")][0]
    str2 = 'output/'+[file for file in files_names if file.startswith("model_")][0]
    #str2 = 'output/'+find('Data_', 'output/')
    #with open(str2, 'r') as f:
    #    args = json.load(f) # dodo
    #args = args['hyper-parameters']
    script = "python test_stanford_networks.py --model_weights_path {} --args_file {} --dataset_path {}".format(str1, str2, data_path)

    script = script.split()    
    subprocess.run(script)
    os.rename('output', current_folder_name)
    
