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

# [425, 191, 128, 128]	[1, 1, 1, 2, 1, 1, 2]	[7, 7, 6, 4, 3, 6]

a = Architecture(hyper_params=hyper_params, hidden_layers=[425, 191, 128, 128], inter_layer_bitwidth=[1, 1, 1, 2, 1, 1, 2], inter_layer_fanin=[7, 7, 6, 4, 3, 6])
label = ""

results = a.final_eval(custom_hyper_params=hyper_params)
print(results)

with open(log_path, 'a', newline='\n') as file:
    file.write(f"{a.hidden_layers}, {a.inter_layer_bitwidth}, {a.inter_layer_fanin}, {round(results[0], 3)}, {results[1]}, '{label}'\n")