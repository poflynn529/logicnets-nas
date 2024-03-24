from architecture import *

log_path = "results/final_eval.txt"

# Hyper-Parameters
hyper_params = {
    "weight_decay": 0.0,
    "batch_size": 1024,
    "epochs": 50,
    "learning_rate": 1e-1,
    "seed": None,
    "checkpoint": None,
    "log_dir": "training-logs",
    "dataset_file": "unsw_nb15_binarized.npz",
    "cuda": True,
}

a = nid_m
a.hyper_params = hyper_params
print(a.compute_utilisation())

#results = a.final_eval(custom_hyper_params=hyper_params)
#print(results)

# with open(log_path, 'a', newline='\n') as file:
#     file.write(f"{a.hidden_layers}, {round(results[0], 3)}, {results[1]}, 'NID-S'\n")

#33, 16, 6, 5