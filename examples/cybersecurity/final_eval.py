from architecture import *

log_path = "results/final_eval.txt"

# Hyper-Parameters
hyper_params = {
    "weight_decay": 0.0,
    "batch_size": 1024,
    "epochs": 200,
    "learning_rate": 2e-2,
    "seed": None,
    "checkpoint": None,
    "log_dir": "training-logs",
    "dataset_file": "unsw_nb15_binarized.npz",
    "cuda": True,
}

# [475, 100, 76, 19, 9, 7]	[1, 1, 2, 3, 2, 3, 3, 1, 3]	[6, 4, 7, 6, 6, 4, 6, 6] 0.901
# [475, 100, 92, 19, 9, 5]	[1, 1, 2, 2, 1, 3, 3, 3, 3]	[6, 4, 7, 6, 6, 4, 6, 6] 0.887


a = Architecture(hyper_params=hyper_params, hidden_layers=[475, 100, 92, 19, 9, 5], inter_layer_bitwidth=[1, 1, 2, 2, 1, 3, 3, 3, 3], inter_layer_fanin=[6, 4, 7, 6, 6, 4, 6, 6])
label = ""

results = a.final_eval(custom_hyper_params=hyper_params)
print(results)

with open(log_path, 'a', newline='\n') as file:
    file.write(f"{a.hidden_layers}, {a.inter_layer_bitwidth}, {a.inter_layer_fanin}, {round(results[0], 3)}, {results[1]}, '{label}'\n")