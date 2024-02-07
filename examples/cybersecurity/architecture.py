import numpy as np
import random
import hashlib
import train

# Architecture class for holding configurable parameters:

class Architecture:

    def __init__(self, hyper_params, hidden_layers, accuracy = -1, utilisation = -1, loss = -1, hash = -1):
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
        estimated_utilisation = self.utilisation
        self.loss = round((estimated_utilisation / self.hyper_params["unity_utilisation"]) * self.hyper_params["utilisation_coeff"] + self.accuracy * self.hyper_params["accuracy_coeff"], 3)

    def evaluate(self):
        self.accuracy = train.main(self)
        print("Training Complete!")
        self.utilisation = self.compute_utilisation()
        self.compute_loss()
        self.hash = Architecture.compute_hash(self.hidden_layers)
    
    @staticmethod
    def compute_hash(layers):
        hash_object = hashlib.md5()
        hash_object.update(str(layers).encode())
        return hash_object.hexdigest()

    def get_csv_data(self):
        print(f"Writing to CSV:   Accuracy: {self.accuracy}, Utilisation: {self.utilisation}, Loss: {self.loss}, Layers: {self.hidden_layers}")
        return [self.accuracy, self.utilisation, self.loss, self.hidden_layers, self.hash]

    
nid_m_arch = [593, 256, 128, 128]