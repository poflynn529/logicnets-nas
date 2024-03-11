# NAS via SDG

import csv
import random
import torch
import numpy as np
from deap import base, creator, tools, algorithms

from architecture import Architecture, nid_m_arch

# Hyper-Parameters
hyper_params = {
    # NAS
    "mode"              : "evo",             # NAS algorithm type. Can be either gradient-based ("grad") or evolutionary-based ("evo").
    "utilisation_coeff" : 0.4,               # Weight of the loss function towards resource utilisation.
    "accuracy_coeff"    : 0.6,               # Weight of the loss function towards accuracy.
    "unity_utilisation" : 16000,             # LUT utilisation considered "nominal" to normalise loss function input.
    "max_iterations"    : 100,               # Maximum amount of architecture explorations per execution of this script
    "target_accuracy"   : 0.92,              # Target accuracy for loss function. After this accuracy is reached, additional accuracy improvements do not reduce the loss.
    "clear_logs"        : True,              # Clear the log files.
    "log_file_path"     : "nas_runs/v2.csv", # NAS Log File Path

    # Training
    "weight_decay": 0.0,
    "batch_size": 1024,
    "epochs": 100,
    "learning_rate": 1e-1,
    "seed": 196,
    "checkpoint": None,
    "log_dir": "training-logs",
    "dataset_file": "unsw_nb15_binarized.npz",
    "cuda": True,
}

# GPU Config
print(f"[NAS] Use CUDA Parameter: {hyper_params['cuda']}")

if torch.cuda.is_available():
    print("[NAS] CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("[NAS] CUDA not available.")

def gradient_search():
    # Compute the numerical gradient of the loss function with respect to the parameter between two architectures.
    def gradient(a1, a2):

        grad = []
        direction = []

        # Gradient will always be negetive for a reducing loss function.
        # Direction indicates which way the parameter moved to get the corresponding loss function shift.

        for i in range(len(a1.hidden_layers)):
            
            # Gradient
            if a1.loss != a2.loss:
                grad.append( (a1.loss - a2.loss) / abs(int(a1.hidden_layers[i]) - int(a2.hidden_layers[i])) ) # Rate of change of the last iteration with respect to the loss function.
            else:
                grad.append(0.0)

            # Direction - This is messy, need to change to some sort of vector / matrix / tuple
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
    
    row_count = 0
    prev_architectures = set()
    a1 = None
    a2 = None

    # Read in the CSV data.
    with open(hyper_params["log_file_path"], 'r') as file:
        
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
            a2 = Architecture.load_from_csv(hyper_params, a2_row, csv_index)

        if a1_row != None:
            a1 = Architecture.load_from_csv(hyper_params, a1_row, csv_index)

        print(f"[NAS] Previously evaluated architectures loaded: {row_count}")

    # We need at least two datapoints to perform the gradient computation:
    if row_count == 0:

        arch = Architecture(hyper_params, nid_m_arch) # Create an instance of the NID-M architecture outlined in the LogicNets paper.
        arch.evaluate()
        print(arch.get_csv_data())
        append_to_csv(hyper_params["log_file_path"], arch.get_csv_data())
        prev_architectures.add(arch.hash)
        
        a1 = arch
        row_count += 1

    if row_count == 1:

        new_layers = random_dx(a1.hidden_layers, allow_no_change=False)
        arch = Architecture(hyper_params, new_layers)
        arch.evaluate()
        append_to_csv(hyper_params["log_file_path"], arch.get_csv_data())
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
            
            new_hash = a1.hash
            while new_hash in prev_architectures:

                for i, layer in enumerate(new_layers): 
                    
                    # This recent change caused the loss function to increase, therefore we go in the opposite direction.
                    if grad_layers[i] > 0:
                        new_layers[i] -= direction[i]
                        
                    # This recent change caused the loss function to decrease, therefore we continue in that direction.
                    elif grad_layers[i] <= 0:
                        new_layers[i] += direction[i]

                new_hash = Architecture.compute_hash(new_layers)
            
            # Create new architecture with the parameters and evaluate.
            arch = Architecture(hyper_params, new_layers)
            arch.evaluate()
            append_to_csv(hyper_params["log_file_path"], arch.get_csv_data())
            prev_architectures.add(arch.hash)

            a2 = a1
            a1 = arch


        print("Search completed! Exiting...")

    # Error condition.
    else:
        print(f"ERROR: Invalid row count: {row_count}, Exiting...")

def genetic_search():
    # Define the fitness criterion - weight is negetive since we want to minimise the loss function.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator: define ranges for each gene
    toolbox.register("attr_int", random.randint, 0, 700)  # The "attr_int" alias now calls random.randint and passes (0, 700) as input arguments.

    # This line creates an individual and uses the "attr_int" function to add 10 genes.
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10) 
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=20) # Create a random population of size 50.

    def evaluate(individual):
        # Convert the individual back to an architecture.
        # Train it, evaluate it, and return the loss metric as fitness.
        arch = Architecture(hyper_params, individual)
        arch.evaluate()
        
        return (arch.loss)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)  # Crossover
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=500, indpb=0.2)  # Mutation
    toolbox.register("select", tools.selTournament, tournsize=3)

    ### Main Evolutionary Code ###

    pop = toolbox.population(n=20)  # Initialize population
    hof = tools.HallOfFame(3)  # Hall of Fame to store the best individual

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)

    print(f"\n\n[pop]: {pop}\n\n")
    print(f"[log]: {log}\n\n")
    print(f"[hof]: {hof}\n\n")


    # # Read in the CSV data.
    # with open(log_file_path, 'r') as file:

    #     reader = csv.reader(file)
    #     csv_index = list_to_dict(next(reader)) # Generate a dictionary with list index of each csv header.

    #     for row in reader:
    #         row_count += 1
    #         prev_architectures.add(row[-1]) # Add the hash to a map to check for it later.
    #         # Add to queue

    #     # Generate initial searches.
    #     if row_count <= pop_size - 1:
            
    #         while row_count != pop_size:
    #             # Create and evalutate random.
    #             pop_size += 1

    #     for i in max_generations:
    #         # Perform evolution, training and logging.
    #         continue

# Main function
if __name__ == "__main__":

    mode = hyper_params["mode"]

    if mode == "grad":
        gradient_search()
    elif mode == "evo":
        genetic_search()
    else:
        print(f"[NAS] Invalid search mode '{mode}'. Exiting...")
    


### Notes

# Step 1: Compute the loss function
# Step 2: Compute the derivative (ratio) of the loss function wrt each parameter
# Step 3: Nudge the parameters by a small amount to minimise the loss function
# Step 4: Continue loop for required epochs.
# Step 5: Log each permutation since we don't want to try something twice.
#         Compute the MD5 hash and store in a binary tree to quickly check for recurrent architectures.


# Step 6: Add an evolutionary component to quickly prototype a diverse array of architectures.
#         SDG can then be used to fine tune the winning architectures.


# Step 7: Look at proxy networks. Keep going until the damn thing has to be submitted.