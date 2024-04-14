# This file is for creating the model weights necessary for synthesis!

from architecture import Architecture
from train import main

hyper_params = {
    "batch_size": 1024,
    "checkpoint": None,
    "cuda": True,
    "dataset_file": "unsw_nb15_binarized.npz",
    "epochs": 200,
    "learning_rate": 2e-2,
    "log_dir": "ga-nid-l",
    "seed": None,
    "weight_decay": 0.0,
}

arch = Architecture(hyper_params=hyper_params, hidden_layers=[475, 100, 76, 19, 9, 7], inter_layer_bitwidth=[1, 1, 2, 3, 2, 3, 3, 1, 3], inter_layer_fanin=[6, 4, 7, 6, 6, 4, 6, 6])

main(arch)