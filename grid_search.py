import json
import os
from copy import deepcopy
import itertools
from main import main, parse_args
from utils import get_stats
from utils import *

def load_config(path="./grid_search_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def run_experiments(args):
    res = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        _, acc2, _ = main(args, i , False)
        res.append(acc2)

    mean, err_bd = get_stats(res, conf_interval=True)
    return mean, err_bd


def grid_search(config: dict):
    args = parse_args()
    cnt = save_cnt = 0
    best_acc, err_bd = 0.0, 0.0
    best_args = vars(args)
    best_args2 = vars(args)
    best_acc2, err_bd2 = 0.0, 0.0
    if args.feat_type != 'identity_feat':
        config.pop('k')
    keys = list(config.keys())
    values = [config[key] for key in keys]
    combinations = list(itertools.product(*values))
    ans = 10000000000
    ans2 = 10000000000
    os.makedirs(args.output_path+"/grid_search/", exist_ok=True)

    if args.data_type == 'regression':
        for combination in combinations:
            param_dict = dict(zip(keys, combination))
            for key, value in param_dict.items():
                setattr(args, key, value)
            acc, bd = run_experiments(args)
            cnt += 1
            if acc < best_acc:
                best_acc = acc
                err_bd = bd
                best_args = deepcopy(vars(args))
                save_cnt = cnt
                ans = args.num_layers * args.hidden_dim
            elif acc == best_acc and args.num_layers * args.hidden_dim < ans:
                best_args = deepcopy(vars(args))
                save_cnt = cnt
                ans = args.num_layers * args.hidden_dim
                                
            if acc <= 0.01 and args.num_layers * args.hidden_dim < ans2:
                best_acc2 = acc
                best_args2 = deepcopy(vars(args))            
                ans2 = args.num_layers * args.hidden_dim
    else:
        for combination in combinations:
            param_dict = dict(zip(keys, combination))
            for key, value in param_dict.items():
                setattr(args, key, value)
            acc, bd = run_experiments(args)
            cnt += 1
            if acc > best_acc:
                best_acc = acc
                err_bd = bd
                best_args = deepcopy(vars(args))
                save_cnt = cnt
                ans = args.num_layers * args.hidden_dim
            elif acc == best_acc and args.num_layers * args.hidden_dim < ans:
                best_args = deepcopy(vars(args))
                save_cnt = cnt
                ans = args.num_layers * args.hidden_dim
                                
            if acc >= 0.95 and args.num_layers * args.hidden_dim < ans2:
                best_acc2 = acc
                best_args2 = deepcopy(vars(args))            
                ans2 = args.num_layers * args.hidden_dim

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #args.output_path = "output/model_{}_{}.log".format(save_cnt,args.dataset)
    result = {
        "params": best_args,
        "params2": best_args2,
        "result": "{:.4f}({:.4f})".format(best_acc, err_bd),
        "result2": "{:.4f}({:.4f})".format(best_acc2, err_bd2),
    }
    with open("{}/model.log".format(args.output_path), "w") as f:
        json.dump(result, f, sort_keys=True, indent=4)


grid_search(load_config())
