# NAS via SDG

import csv
import random

from architecture import Architecture, nid_m_arch

# All run data will be stored here.
log_file_path = "nas_runs/v1.csv"

# Hyper-Parameters
hyper_params = {
    # NAS
    "utilisation_coeff" : 0.5,   # Weight of the loss function towards resource utilisation.
    "accuracy_coeff"    : 0.5,   # Weight of the loss function towards accuracy.
    "unity_utilisation" : 16000, # LUT utilisation considered "nominal" to normalise loss function input.
    "max_iterations"    : 100,  # Maximum amount of architecture explorations per execution of this script

    # Training
    "weight_decay": 0.0,
    "batch_size": 1024,
    "epochs": 100,
    "learning_rate": 1e-1,
    "seed": 196,
    "checkpoint": None,
}

# Compute the gradient with the respect to the loss function between two architectures.
def gradient(a1, a2):

    grad = []
    direction = []

    # Gradient will always be negetive for a reducing loss function.
    # Direction indicates which way the parameter moved to get the corresponding loss function shift.

    for i in range(len(a1.hidden_layers)):
        
        # Gradient
        if a1.loss != a2.loss:
            grad.append( abs(a1.hidden_layers[i] - a2.hidden_layers[i]) / (a1.loss - a2.loss) ) # Rate of change of the last iteration with respect to the loss function.
        else:
            grad.append(0.0)

        # Direction
        if a1.hidden_layers[i] - a2.hidden_layers[i] > 0:
            direction.append(1)
        elif a1.hidden_layers[i] - a2.hidden_layers[i] < 0:
            direction.append(-1)
        else:
            direction.append(0)

    return grad, direction

# Append a list to a csv file.
def append_to_csv(file_name, data):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Randomly increment or decrement an array.
def random_dx(arr, type="int", dx=1, allow_no_change=True):
    if type not in ["int", "float"]:
        raise ValueError("Type must be 'int' or 'float'")
    
    options = None

    if allow_no_change:
        options = [-dx, dx, 0]
    else:
        options = [-dx, dx]

    result = []
    for item in arr:
        if type == "int" and not isinstance(item, int):
            raise ValueError("All elements must be integers")
        elif type == "float" and not isinstance(item, float):
            raise ValueError("All elements must be floats")

        change = random.choice(options)
        new_value = item + change

        result.append(new_value)

    return result

# Convert list of strings to dictionary with the value being the index.
def list_to_dict(lst):
    return {string: index for index, string in enumerate(lst)}

# Main function
if __name__ == "__main__":

    row_count = 0
    prev_architectures = set()
    a1 = None
    a2 = None

    # Read in the CSV data.
    with open(log_file_path, 'r') as file:
        
        a1_row = None
        a2_row = None

        reader = csv.reader(file)
        csv_index = list_to_dict(next(reader)) # Generate a dictionary with list index of each csv header.

        for row in reader:
            a2_row = a1_row
            a1_row = row
            row_count += 1
            prev_architectures.add(row[-1]) # Add the hash to a map to check for it later.

        if a2_row != None:
            a2 = Architecture(hyper_params, a2_row[csv_index["layers"]], a2_row[csv_index["accuracy"]], a2_row[csv_index["utilisation"]], a2_row[csv_index["loss"]], a2_row[csv_index["hash"]])

        if a1_row != None:
            a1 = Architecture(hyper_params, a1_row[csv_index["layers"]], a1_row[csv_index["accuracy"]], a1_row[csv_index["utilisation"]], a1_row[csv_index["loss"]], a1_row[csv_index["hash"]])

        print(f"Previously evaluated architectures loaded: {row_count}")

    # We need at least two datapoints to perform the gradient computation:
    if row_count == 0:

        arch = Architecture(hyper_params, nid_m_arch) # Create an instance of the NID-M architecture outlined in the LogicNets paper.
        arch.evaluate()
        print(arch.get_csv_data())
        append_to_csv(log_file_path, arch.get_csv_data())
        prev_architectures.add(arch.hash)
        
        a1 = arch
        row_count += 1

    if row_count == 1:

        new_layers = random_dx(a1.hidden_layers, allow_no_change=False)
        arch = Architecture(hyper_params, new_layers)
        arch.evaluate()
        append_to_csv(log_file_path, arch.get_csv_data())
        prev_architectures.add(arch.hash)

        a2 = a1
        a1 = arch
        row_count += 1
    
    # At this stage we should have at least 2 architectures and can compute the gradient.
    if row_count >= 2:

        for i in range(1, hyper_params["max_iterations"]):
            grad_layers, direction = gradient(a1, a2)

            # Now we adjust each parameter according to the gradient.
            new_layers = a1.hidden_layers.copy()
            
            for i, layer in enumerate(new_layers):
                hash = a1.hash
                while hash in prev_architectures:
                    # This recent change caused the loss function to increase, therefore we go in the opposite direction.
                    if grad_layers[i] > 0:
                        new_layers[i] -= direction[i]
                        
                    # This recent change caused the loss function to decrease, therefore we continue in that direction.
                    elif grad_layers[i] < 0:
                        new_layers[i] += direction[i]
            
            # Create new architecture with the parameters and evaluate.
            arch = Architecture(hyper_params, new_layers)
            arch.evaluate()
            append_to_csv(log_file_path, arch.get_csv_data())
            prev_architectures.add(arch.hash)

            a2 = a1
            a1 = arch


        print("Search completed! Exiting...")

    # Error condition.
    else:
        print(f"ERROR: Invalid row count: {row_count}, Exiting...")


# Step 1: Compute the loss function
# Step 2: Compute the derivative (ratio) of the loss function wrt each parameter
# Step 3: Nudge the parameters by a small amount to minimise the loss function
# Step 4: Continue loop for required epochs.
# Step 5: Log each permutation since we don't want to try something twice.
#         Compute the MD5 hash and store in a binary tree to quickly check for recurrent architectures.


# Step 6: Add an evolutionary component to quickly prototype a diverse array of architectures.
#         SDG can then be used to fine tune the winning architectures.


# Step 7: Look at proxy networks. Keep going until the damn thing has to be submitted.

# New comment for test.