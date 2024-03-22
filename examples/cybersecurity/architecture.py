import numpy as np
import random
import hashlib
import train
import ast

# Hyper-Parameters
hyper_params = {
    # NAS General
    "mode"                : "evo",                 # NAS algorithm type. Can be either gradient-based ("grad") or evolutionary-based ("evo").
    "utilisation_coeff"   : 0.2,                   # Weight of the loss function towards resource utilisation.
    "accuracy_coeff"      : 0.8,                   # Weight of the loss function towards accuracy.
    "unity_utilisation"   : 1105,                 # LUT utilisation considered "nominal" to normalise loss function input.
    "target_accuracy"     : 0.92,                  # Target accuracy for loss function. After this accuracy is reached, additional accuracy improvements do not reduce the loss.

    # Gradient NAS Specific
    "grad_clear_logs"     : False,                 # Clear the log files.
    "grad_log_file_path"  : "nas_runs/grad_0.csv", # NAS Log File Path
    "grad_max_iterations" : 100,                   # Maximum amount of architecture explorations per execution of this script

    # Evolutionary NAS Specific
    "pop_size"            : 10,
    "max_generations"     : 40,
    "crossover_prob"      : 0.5,
    "mutation_prob"       : 0.4,
    "tournsize"           : 2,
    "max_layer_size"      : 600,
    "min_layer_size"      : 5,
    "max_bitwidth"        : 3,
    "min_bitwidth"        : 1,   
    "evo_log_file_path"   : "nas_runs/evo_0.txt",
    "evo_pickle_path"     : "nas_pickle/evo_0.pkl",

    # Training
    "weight_decay": 0.0,
    "batch_size": 1024,
    "epochs": 25,
    "learning_rate": 1e-1,
    "seed": None,
    "checkpoint": None,
    "log_dir": "training-logs",
    "dataset_file": "unsw_nb15_binarized.npz",
    "cuda": True,
}

# Architecture class for holding configurable parameters:

class Architecture:

    def __init__(self, hyper_params, hidden_layers, accuracy = None, utilisation = None, loss = None, hash = None):
        self.hyper_params = hyper_params
        self.hidden_layers = hidden_layers
        self.input_bitwidth = 1
        self.hidden_bitwidth = 2
        self.output_bitwidth = 2
        self.input_fanin = 7
        self.hidden_fanin = 7
        self.output_fanin = 7
        self.accuracy = accuracy
        self.utilisation = utilisation
        self.loss = loss
        self.hash = hash

    # Compute utilisation based on the layer sizes and input and output bitwidths
    def compute_utilisation(self):

        # LUT cost of NEQ based on LogicNets paper where X and Y corresponding to the bitwidth of the input
        # and output respectively.
        def lut_cost(X, Y): 
            return round((float(Y) / 3) * (np.power(2.0, (float(X) - 4)) - np.power(-1.0, float(X))))

        util = 0
        for layer in self.hidden_layers:
            util += layer * lut_cost(self.input_bitwidth, self.output_bitwidth)

        return util

    # Compute the loss of a particular architecture.
    def compute_loss(self):

        utilisation_loss = (self.utilisation / self.hyper_params["unity_utilisation"])
        
        if self.accuracy > self.hyper_params["target_accuracy"]:
            accuracy_loss = 0
            #print(f"[DEBUG] Target Accuracy Acheived, AccLoss: {accuracy_loss}")
        else:
            accuracy_loss = (self.hyper_params["target_accuracy"] - self.accuracy)
            #print(f"[DEBUG] AccLoss: {accuracy_loss}")

        return round(utilisation_loss * self.hyper_params["utilisation_coeff"] + accuracy_loss * self.hyper_params["accuracy_coeff"], 3)

    def evaluate(self):
        self.accuracy = np.mean(train.main(self)) / 100
        #print(f"[DEBUG] Final Validation Accuracy: {self.accuracy}")
        self.utilisation = self.compute_utilisation()
        self.loss = self.compute_loss()
        self.hash = Architecture.compute_hash(self.hidden_layers)
        
    @staticmethod
    def compute_hash(layers):
        hash_object = hashlib.md5()
        hash_object.update(str(layers).encode())
        return hash_object.hexdigest()

    def get_csv_data(self):
        #print(f"[DEBUG] Writing to CSV:   Accuracy: {self.accuracy}, Utilisation: {self.utilisation}, Loss: {self.loss}, Layers: {self.hidden_layers}")
        return [self.accuracy, self.utilisation, self.loss, self.hidden_layers, self.hash]

    @staticmethod
    def load_from_csv(hyper_params, row, csv_index):
        return Architecture(hyper_params, hidden_layers=ast.literal_eval(row[csv_index["layers"]]), accuracy=float(row[csv_index["accuracy"]]), utilisation=int(row[csv_index["utilisation"]]), loss=float(row[csv_index["loss"]]), hash=row[csv_index["hash"]])

nid_m = [593, 256, 128, 128]
nid_m_comp = [593, 256, 49, 7]
nid_s = [593, 100]
nid_s_comp = [49, 7]
nid_l = [593, 100, 100, 100]
nid_l_comp = [593, 100, 25, 5]

nid_s_arch = Architecture(hyper_params=hyper_params, hidden_layers=nid_s, )
nid_m_arch = Architecture(hyper_params=hyper_params, hidden_layers=nid_m, )


