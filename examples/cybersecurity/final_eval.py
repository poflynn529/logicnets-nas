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

a = Architecture(hyper_params=hyper_params, hidden_layers=[508, 75, 24, 16], inter_layer_bitwidth=[1, 2, 1, 2, 2, 2, 2], inter_layer_fanin=[5, 6, 5, 4, 5, 7])
label = ""

results = a.final_eval(custom_hyper_params=hyper_params)
print(results)

with open(log_path, 'a', newline='\n') as file:
    file.write(f"{a.hidden_layers}, {a.inter_layer_bitwidth}, {a.inter_layer_fanin}, {round(results[0], 3)}, {results[1]}, '{label}'\n")