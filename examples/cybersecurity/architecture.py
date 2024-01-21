import numpy as np
import time

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
        self.loss = (estimated_utilisation / self.hyper_params["unity_utilisation"]) * self.hyper_params["utilisation_coeff"] + self.accuracy *["accuracy_coeff"]

    def evaluate(self):
        time.sleep(1)
        self.accuracy = 0.92
        self.utilisation = 16500
        self.compute_loss(self)

    
nid_m_arch = [593, 256, 128, 128]