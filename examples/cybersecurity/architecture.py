import numpy as np
import random
import hashlib

# Architecture class for holding configurable parameters:

class Architecture:

    def __init__(self, hyper_params, hidden_layers, accuracy = -1, utilisation = -1, loss = -1, hash = -1):
        self.hyper_params = hyper_params
        self.hidden_layers = hidden_layers
        self.input_bidwith = 1
        self.hidden_bitwidth = 2
        self.output_bitwidth = 2
        self.input_fanin = 7
        self.hidden_fanin = 7
        self.output_fanin = 7
        self.accuracy = accuracy
        self.utilisation = utilisation
        self.loss = loss
        self.hash = hash

    # Compute the loss of a particular architecture.
    def compute_loss(self):

        # LUT cost of NEQ based on LogicNets paper where X and Y corresponding to the bitwidth of the input
        # and output respectively.
        def lut_cost(X, Y): 
            return (Y / 3) * (np.pow(2, (X - 4)) - np.pow(-1, X))

        estimated_utilisation = self.utilisation
        self.loss = round((estimated_utilisation / self.hyper_params["unity_utilisation"]) * self.hyper_params["utilisation_coeff"] + self.accuracy * self.hyper_params["accuracy_coeff"], 3)

    def evaluate(self):
        self.accuracy = round(0.92 + random.uniform(-0.1, 0.02), 3)
        self.utilisation = 16500 + random.randint(-3500, 14000)
        self.compute_loss()
        self.hash = self.compute_hash()
    
    def compute_hash(self):
        hash_object = hashlib.md5()
        hash_object.update(str(self.hidden_layers).encode())
        return hash_object.hexdigest()

    def get_csv_data(self):
        return [self.accuracy, self.utilisation, self.loss, self.hidden_layers, self.hash]

    
nid_m_arch = [593, 256, 128, 128]