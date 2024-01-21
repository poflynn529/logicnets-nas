# NAS via SDG

import numpy as np

# All run data will be stored here.
log_file_path = 

# NAS Hyper-Parameters
hyper_params = {
    "utilisation_coeff" : 0.5,  # Weight of the loss function towards resource utilisation.
    "accuracy_coeff"    : 0.5,  # Weight of the loss function towards accuracy.
    "unity_utilisation" : 16000 # LUT utilisation considered "nominal" to normalise loss function input.
}

def loss_function(architecture, accuracy):

    # LUT cost of NEQ based on LogicNets paper where X and Y corresponding to the bitwidth of the input
    # and output respectively.
    def lut_cost(X, Y): 
        return (Y / 3) * (np.pow(2, (X - 4)) - np.pow(-1, X))

    estimated_utilisation = sum(architecture)
    return (estimated_utilisation / hyper_params["unity_utilisation"]) * hyper_params["utilisation_coeff"] + accuracy *["accuracy_coeff"]

if __name__ == "__main__":

    if 

# Step 1: Compute the loss function
# Step 2: Compute the derivative (ratio) of the loss function wrt each parameter
# Step 3: Nudge the parameters by a small amount to minimise the loss function
# Step 4: Continue loop for required epochs.
# Step 5: Log each permutation since we don't want to try something twice.
#         Compute the MD5 hash and store in a binary tree to quickly check for recurrent architectures.


# Step 6: Add an evolutionary component to quickly prototype a diverse array of architectures.
#         SDG can then be used to fine tune the winning architectures.


# Step 7: Look at proxy networks. Keep going until the damn thing has to be submitted.