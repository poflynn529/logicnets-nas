import numpy as np
import random
import hashlib
import train
import ast

# Architecture class for holding configurable parameters:

class ArchitectureOld:

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
            util += layer * lut_cost(self.hidden_bitwidth*self.input_fanin, self.output_bitwidth)

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

    def final_eval(self, custom_hyper_params):
        return np.max(train.main(self, custom_hyper_params)) / 100, self.compute_utilisation()
        
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

class Architecture:

    def __init__(self, hyper_params, hidden_layers, inter_layer_bitwidth, inter_layer_fanin, accuracy = None, utilisation = None, loss = None, hash = None):

        # Valid input checks
        # The +3 derives from Hidden Layers + Input Layer + Output Layer + 1
        if len(inter_layer_bitwidth) != len(hidden_layers) + 3:
            raise(ValueError(f"Invalid Inter-Layer Bitwidth Input:\tHidden Layers: {len(hidden_layers)}, Bitwidth Vector Length: {len(inter_layer_bitwidth)}"))
        
        # The +2 derives from Hidden Layers + Input Layer + Output Layer
        if len(inter_layer_fanin) != len(hidden_layers) + 2: 
            raise(ValueError(f"Invalid Inter-Layer Fan-In Input:\tHidden Layers: {len(hidden_layers)}, Fan-In Vector Length: {len(inter_layer_fanin)}"))

        self.hyper_params = hyper_params
        self.hidden_layers = hidden_layers
        self.inter_layer_bitwidth = inter_layer_bitwidth
        self.inter_layer_fanin = inter_layer_fanin
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

        # Input Layer
        util += 593 * lut_cost(self.inter_layer_fanin[0] * self.inter_layer_bitwidth[0], self.inter_layer_bitwidth[1])
        
        # Hidden Layers
        for i, layer in enumerate(self.hidden_layers):
            util += layer * lut_cost(self.inter_layer_fanin[i + 1]*self.inter_layer_bitwidth[i + 1], self.inter_layer_bitwidth[i + 2])

        # Output Layer
        util += lut_cost(self.inter_layer_fanin[-1]*self.inter_layer_bitwidth[-2], self.inter_layer_bitwidth[-1])

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
        self.hash = Architecture.compute_hash(self.hidden_layers, self.inter_layer_bitwidth, self.inter_layer_fanin)

    def final_eval(self, custom_hyper_params):
        return np.max(train.main(self, custom_hyper_params)) / 100, self.compute_utilisation()
        
    @staticmethod
    def compute_hash(hidden_layers, inter_layer_bitwidth, inter_layer_fanin):
        hash_object = hashlib.md5()
        hash_object.update(str([hidden_layers, inter_layer_bitwidth, inter_layer_fanin]).encode())
        return hash_object.hexdigest()

    def get_csv_data(self):
        #print(f"[DEBUG] Writing to CSV:   Accuracy: {self.accuracy}, Utilisation: {self.utilisation}, Loss: {self.loss}, Layers: {self.hidden_layers}")
        return [self.accuracy, self.utilisation, self.loss, self.hidden_layers, self.hash]

    @staticmethod
    def load_from_csv(hyper_params, row, csv_index):
        return Architecture(hyper_params, hidden_layers=ast.literal_eval(row[csv_index["layers"]]), accuracy=float(row[csv_index["accuracy"]]), utilisation=int(row[csv_index["utilisation"]]), loss=float(row[csv_index["loss"]]), hash=row[csv_index["hash"]])

nid_m = Architecture(None, [593, 256, 128, 128], [1, 2, 2, 2, 2, 2, 2], [7, 7, 7, 7, 7, 7])
nid_m_comp = [593, 256, 49, 7]
nid_s = [593, 100]
nid_s_comp = [49, 7]
nid_l = [593, 100, 100, 100]
nid_l_comp = [593, 100, 25, 5]
