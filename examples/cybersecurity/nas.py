# NAS via SDG

import csv
import random
import pickle
import os
import copy
import yaml

import torch
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from architecture import Architecture

# Hyper-Parameters
with open('hyper_params.yaml', 'r') as file:
    # Use safe_load() to load the file's contents into a dictionary
    hyper_params = yaml.safe_load(file)

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

        arch = Architecture(hyper_params) # Create an instance of the NID-M architecture outlined in the LogicNets paper.
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

    # Allow checkpointing using pickle so that we can stop/start the algorithm.
    def save_checkpoint(df, filename=hyper_params["evo_pickle_path"]):
        print(f"[NAS] Saving checkpoint to {filename}.")
        df.to_pickle(filename)

    def load_checkpoint(df_log_path=hyper_params["evo_pickle_path"]):
        if os.path.exists(df_log_path):
            print(f"[NAS] Loading checkpoint from {df_log_path}.")
            df = pd.read_pickle(df_log_path)
        else:
            print("[NAS] Pickle Log does not exist, creating new DataFrame.")
            df = pd.DataFrame(columns=['gen', 'hash', 'hidden_layers', 'inter_layer_bitwidth', 'inter_layer_fanin', 'proxy_utilisation', 'proxy_accuracy', 'fitness', 'evaluated', 'training_epochs', 'learning_rate'])

        return df
    
    def load_seed(seed_path):
        print("[NAS] Loading seed information...")
        df = pd.read_pickle(seed_path)
        for idx in df.index.tolist():
            df.loc[idx, "hash"] = Architecture.compute_hash(df.loc[idx, "hidden_layers"], df.loc[idx, "inter_layer_bitwidth"], df.loc[idx, "inter_layer_fanin"])
        return df

    def save_log(gen, pop, log, filename=hyper_params["evo_log_file_path"]):
        with open(filename, 'a', newline='\n') as file:
            print("Writing Log...")
            file.write(f"\n###### Generation {gen} ######\n\n")
            for i, individual in enumerate(pop):
                file.write(f"INDV {i}:\t{str(individual)}, Fitness: {individual.fitness.values}, Accuracy: {round(individual.proxy_accuracy, 3)}, Utilisation: {individual.utilisation}\n")
            file.write("\nLog:\n\n")
            file.write(str(log))
            file.write("\n")

    def save_hof(hof, filename=hyper_params["evo_log_file_path"]):
        with open(filename, 'a', newline='\n') as file:
            file.write(f"\n###### Hall of Fame ######\n\n")
            file.write(f"{hof}\n\n")

    def custom_deepcopy(pd_obj, obj_columns):
        # The list objects inside the dataframes/series will not be correctly deepcopied causing dataframe/series corruption.
        # This function fixes that by manually deepcopying the python objects stored inside the dataframe.
        pd_obj_copy = copy.deepcopy(pd_obj)

        if type(pd_obj) == type(pd.DataFrame(dtype=object)):
            for obj_name in obj_columns:
                for idx in pd_obj_copy.index.tolist():
                    pd_obj_copy.at[idx, obj_name] = copy.deepcopy(pd_obj.loc[idx, obj_name])

        elif type(pd_obj) == type(pd.Series(dtype=object)):
            for idx in obj_columns:
                    pd_obj_copy.at[idx] = copy.deepcopy(pd_obj.loc[idx])

        else:
            raise(TypeError(f"custom_deepcopy() does not support type: {type(pd_obj)}. Object len: {len(pd_obj)}. Supported types: {type(pd.DataFrame())}, {type(pd.Series())}."))
        
        return pd_obj_copy

    def validate(df):
        # Ensure architectures with the same hash have the same parameters.
        groups = df.drop("gen", axis=1).groupby("hash")
        for name, group in groups:
            for idx in group.index:
                if not (group.loc[idx] == group.iloc[0]).all():
                    save_checkpoint(df)
                    raise ValueError(f"Hash {group['hash'].iloc[0]} contains different architectures!")

        # Ensure dataframe is maintaining a consistent index.
        if not df.index.is_unique:
            save_checkpoint(df)
            raise(ValueError(f"Index values are not unique!"))
    
    def evaluate(df):

        not_evaluated_index = df[df["evaluated"] == False].index.tolist()
        print(f"[NAS-Evaluate] {len(not_evaluated_index)} evaluation(s) to do at indexes: {not_evaluated_index}")

        for idx in not_evaluated_index:
            
            df.loc[idx, "hash"] = Architecture.compute_hash(df.loc[idx, "hidden_layers"], inter_layer_bitwidth=df.loc[idx, "inter_layer_bitwidth"], inter_layer_fanin=df.loc[idx, "inter_layer_fanin"])

            if df.loc[idx, "hash"] in df[df["evaluated"] == True]["hash"].values:
                matched_architecture = df[(df["hash"] == df.loc[idx, "hash"]) & (df["evaluated"] == True)].iloc[0]
                #print(matched_architecture)
                #print(df.loc[idx])
                df.loc[idx, "proxy_utilisation"] = matched_architecture["proxy_utilisation"]
                df.loc[idx, "proxy_accuracy"] = matched_architecture["proxy_accuracy"]
                df.loc[idx, "fitness"] = matched_architecture["fitness"]
                df.loc[idx, "evaluated"] = True
                print(f"[NAS-Evaluate] Found matching architecture for {df.loc[idx, 'hash']} (Index: {idx}).")
            else:
                arch = Architecture(hyper_params, hidden_layers=df.loc[idx, "hidden_layers"], inter_layer_bitwidth=df.loc[idx, "inter_layer_bitwidth"], inter_layer_fanin=df.loc[idx, "inter_layer_fanin"])
                arch.evaluate()
                
                df.loc[idx, "proxy_utilisation"] = arch.utilisation
                df.loc[idx, "proxy_accuracy"] = arch.accuracy
                df.loc[idx, "fitness"] = arch.loss
                df.loc[idx, "evaluated"] = True
                df.loc[idx, "training_epochs"] = hyper_params["epochs"]
                df.loc[idx, "learning_rate"] = hyper_params["learning_rate"]

    
    # Returns a DEAP Individual object that contains the parameters. 
    def create_random_individual(hyper_params, gen):

        ### Generate Layer Size ###
        hidden_layers=[random.randint(hyper_params["min_layer_size"], hyper_params["max_layer_size"])]

        # We want our layer size as a descending list:
        for i in range(hyper_params["hidden_layers"] - 1):
            hidden_layers.append(random.randint(hyper_params["min_layer_size"], hidden_layers[i]))
        ### End Generate Layer Size ###

        ### Generate Bitwidth ###
        inter_layer_bitwidth = []
        for i in range(hyper_params["hidden_layers"] + 3):
            inter_layer_bitwidth.append(random.randint(hyper_params["min_bitwidth"], hyper_params["max_bitwidth"]))
        ### End Generate Bitwidth ###
            
        ### Generate Fanin ###
        inter_layer_fanin = []
        for i in range(hyper_params["hidden_layers"] + 2):
            inter_layer_fanin.append(random.randint(hyper_params["min_fanin"], hyper_params["max_fanin"]))
        ### End Generate Fanin ###

        individual = {
            'gen'  : gen,
            'hidden_layers' : hidden_layers,
            'inter_layer_bitwidth' : inter_layer_bitwidth,
            'inter_layer_fanin' : inter_layer_fanin,
            'evaluated' : False
        }

        return individual
    
    # Mutate individual while maintaining descending layer size.
    def mutate(individual):

        if random.random() <= hyper_params["mutation_prob"]:
            hidden_layers = copy.deepcopy(individual["hidden_layers"])
            inter_layer_bitwidth = copy.deepcopy(individual["inter_layer_bitwidth"])
            inter_layer_fanin = copy.deepcopy(individual["inter_layer_fanin"])

            ### Layer Size Mutation ###
            if random.random() <= hyper_params["gene_mutation_prob"]:
                hidden_layers[0] = random.randint(hyper_params["min_layer_size"], hyper_params["max_layer_size"])
                                            
            for i in range(len(hidden_layers) - 1):
                if random.random() <= hyper_params["gene_mutation_prob"]:
                    hidden_layers[i + 1] = random.randint(hyper_params["min_layer_size"], hidden_layers[i])

            # Descending Layer check.
            for i in range(1, hyper_params["hidden_layers"]):
                    if hidden_layers[i] > hidden_layers[i - 1]:
                        hidden_layers[i] = random.randint(hyper_params["min_layer_size"], hidden_layers[i - 1])    
                
            ### End Layer Size Mutation ###
                    
            ### Inter-Layer Bitwidth Mutation ###
            for i in range(len(inter_layer_bitwidth)):
                if random.random() <= hyper_params["gene_mutation_prob"]:
                    inter_layer_bitwidth[i] = random.randint(hyper_params["min_bitwidth"], hyper_params["max_bitwidth"])
            ### End Inter-Layer Bitwidth Mutation ###
                    
            ### Inter-Layer Fanin Mutation ###
            for i in range(len(inter_layer_fanin)):
                if random.random() <= hyper_params["gene_mutation_prob"]:
                    inter_layer_fanin[i] = random.randint(hyper_params["min_fanin"], hyper_params["max_fanin"])
            ### End Inter-Layer Fanin Mutation ###

            individual["hidden_layers"] = hidden_layers
            individual["inter_layer_bitwidth"] = inter_layer_bitwidth
            individual["inter_layer_fanin"] = inter_layer_fanin

        return individual
    
    def serialise_params(pd_obj):

        if type(pd_obj) == type(pd.DataFrame(dtype=object)):
            serial_list = copy.deepcopy(pd_obj["hidden_layers"].iloc[0] + pd_obj["inter_layer_bitwidth"].iloc[0] + pd_obj["inter_layer_fanin"].iloc[0])
            #print(f"[DEBUG] serialise_params() returning: {serial_list} from DataFrame")
        elif type(pd_obj) == type(pd.Series(dtype=object)):
            serial_list = copy.deepcopy(pd_obj["hidden_layers"] + pd_obj["inter_layer_bitwidth"] + pd_obj["inter_layer_fanin"])
            #print(f"[DEBUG] serialise_params() returning: {serial_list} from Series")
        else:
            raise(TypeError(f"serialise_params() does not support type: {type(pd_obj)}. Object len: {len(pd_obj)}. Supported types: {type(pd.DataFrame())}, {type(pd.Series())}."))

        return serial_list
    
    def deserialise_params(list_obj):

        params_dict = {
            "hidden_layers"        : list_obj[0:hyper_params["hidden_layers"]],
            "inter_layer_bitwidth" : list_obj[hyper_params["hidden_layers"]:hyper_params["hidden_layers"]*2 + 3],
            "inter_layer_fanin"    : list_obj[hyper_params["hidden_layers"]*2 + 3:hyper_params["hidden_layers"]*3 + 5],
        }

        return params_dict
    
    # Cross individuals while maintaining descending layer size.
    def crossover(cross_df):

        cross_df = cross_df.reset_index(drop=True) # Index must be reset, otherwise duplicate individuals will be lumped together by the for loop.
        #print(df)
        for i in cross_df.index.tolist():
            if random.random() <= hyper_params["crossover_prob"]:
                indv1 = serialise_params(cross_df.loc[i])
                indv2 = serialise_params(cross_df.sample(1))

                # print(f"[DEBUG] indv1: {indv1}")
                # print(f"[DEBUG] indv2: {indv2}")

                # Perform cross at n randomly chosen points.
                for j in range(int(hyper_params["gene_crossover_prob"] * hyper_params["hidden_layers"] * 3 + 5)):
                    cross_position = random.randint(0, len(indv1) - 1)
                    indv1[cross_position] = indv2[cross_position]

                # Check for descending constraint violation for layer size only.
                for j in range(1, hyper_params["hidden_layers"]):
                    if indv1[j] > indv1[j - 1]:
                        indv1[j] = random.randint(hyper_params["min_layer_size"], indv1[j - 1])

                # print(f"Hidden Layers: {indv1[0:hyper_params['hidden_layers']]}, Existing Value: {df.loc[i, 'hidden_layers']}")
                # print(f"Inter Layer Bitwidth: {indv1[hyper_params['hidden_layers']:hyper_params['hidden_layers']*2 + 3]}, Existing Value: {df.loc[i, 'inter_layer_bitwidth']}")
                # print(f"Inter Layer Fanin: {indv1[hyper_params['hidden_layers']*2 + 3:hyper_params['hidden_layers']*3 + 5]}, Existing Value: {df.loc[i, 'inter_layer_fanin']}")

                # print(f"New Type: {type(indv1[0:hyper_params['hidden_layers']])}, Existing Type: {type(df.loc[i, 'hidden_layers'])}")

                params_dict = deserialise_params(indv1)

                cross_df.at[i, "hidden_layers"] = params_dict["hidden_layers"]
                cross_df.at[i, "inter_layer_bitwidth"] = params_dict["inter_layer_bitwidth"]
                cross_df.at[i, "inter_layer_fanin"] = params_dict["inter_layer_fanin"]

        return cross_df

    def evolvePop(gen_to_evolve, hyper_params):

        new_population = pd.DataFrame(columns=gen_to_evolve.columns.tolist())
        gen_to_evolve["evaluated"] = False
        gen_to_evolve["hash"] = None
        gen_to_evolve["fitness"] = pd.to_numeric(gen_to_evolve["fitness"], errors='coerce') # Workaround for bug in older pandas version.

        ### Tournament ###
        for i in range(hyper_params["pop_size"] - hyper_params["elitism"] - hyper_params["randoms"]):
            # Randomly select a number of individuals from the population and add them for mutation.
            new_population = new_population.append(custom_deepcopy(gen_to_evolve.loc[gen_to_evolve.sample(hyper_params["tourn_size"])["fitness"].idxmin()], ["hidden_layers", "inter_layer_bitwidth", "inter_layer_fanin"]), ignore_index=True)
            new_population.reset_index(drop=True, inplace=True)

        #print(f"\nTournament Selection:\n\n{new_population[['gen', 'hidden_layers', 'inter_layer_bitwidth', 'inter_layer_fanin', 'proxy_utilisation', 'proxy_accuracy', 'fitness']]}\n")
        
        ### Mutation ###
        for idx in new_population.index.tolist():
            new_population.loc[idx] = mutate(custom_deepcopy(new_population.loc[idx], ["hidden_layers", "inter_layer_bitwidth", "inter_layer_fanin"]))

        ### Crossover ###
        new_population = crossover(new_population)

        ### Elitism & Randoms ###
        new_population = new_population.append(custom_deepcopy(gen_to_evolve.nsmallest(hyper_params["elitism"], "fitness"), ["hidden_layers", "inter_layer_bitwidth", "inter_layer_fanin"]), ignore_index=True) # Elitism
        for i in range(hyper_params["randoms"]):
            new_population = new_population.append(create_random_individual(hyper_params=hyper_params, gen=gen_to_evolve["gen"].max()), ignore_index=True) # Randoms

        new_population["gen"] += 1
        new_population["evaluated"] = False

        return new_population

    def eaCustom(hyper_params, df_log_path, ngen, seed_individuals_path=None, text_log_path=None, verbose=False):
        
        df = load_checkpoint(df_log_path)

        # Find the current maximum generation and enter the evolutionary loop.
        if pd.isna(df["gen"].max()):
            start_gen = 0
            # Seed individuals
            if hyper_params["evo_seed_path"] != "None":
                df = load_seed(hyper_params["evo_seed_path"])

            # Create (n - seed) random individuals
            for i in range(hyper_params["pop_size"] - len(df)):
                df = df.append(create_random_individual(hyper_params=hyper_params, gen=start_gen), ignore_index=True)
                
        else:
            start_gen = int(df["gen"].max())
            print(f"[NAS] Previously loaded generations: {start_gen + 1}")
        
        for i in range(start_gen + 1, ngen):
            
            evaluate(df) # Ensure all individuals currently in the population have been evaluated.

            prev_gen = df[df["gen"] == i - 1]
            new_generation = evolvePop(custom_deepcopy(prev_gen, ["hidden_layers", "inter_layer_bitwidth", "inter_layer_fanin"]), hyper_params=hyper_params) # custom_deepcopy must be used here. It is not equivalent to the pandas .copy method or copy.deepcopy!

            df = df.append(new_generation, ignore_index=True) # Copy the last generation and evolve a new one.
            df.reset_index(drop=True, inplace=True)
            
            evaluate(df)

            if verbose:
                print(f"\nNew Population:\n\n{df[df['gen'] == i][['gen', 'hash', 'hidden_layers', 'inter_layer_bitwidth', 'inter_layer_fanin', 'proxy_utilisation', 'proxy_accuracy', 'fitness']]}\n")

            # Perform a validation check.
            validate(df.copy())

            save_checkpoint(df, df_log_path) # Save checkpoint to allow analysis and interrupted execution.

            if verbose:
                current_gen = df[df["gen"] == i]
                print(f"\nGeneration: {i}, FitnessMax: {round(current_gen['fitness'].max(), 4)}, FitnessMin: {round(current_gen['fitness'].min(), 4)}, FitnessAvg: {round(current_gen['fitness'].mean(), 4)}\n")
                print("#############################################################################################\n")

        print("[NAS] Search Complete! Top 10 Architectures:")
        df["fitness"] = pd.to_numeric(df["fitness"])
        print(df.groupby('hash').first().nsmallest(10, 'fitness'))

    # Entry point
    eaCustom(hyper_params=hyper_params, df_log_path=hyper_params["evo_pickle_path"], ngen=hyper_params["max_generations"], verbose=True)
        
    ### Tests ###
    def test_mutate():
        test_series = pd.Series(create_random_individual(gen=0, hyper_params=hyper_params))
        for i in range(1000):
            mutate(test_series)
            for j in range(1, hyper_params["hidden_layers"]):
                if test_series["hidden_layers"][j - 1] < test_series["hidden_layers"][j]:
                    print(f"[TEST] Mutate Failed i = {i}, hidden_layers = {test_series['hidden_layers']}!")

    def test_crossover():
        return
    
    #test_mutate()


# Main function
if __name__ == "__main__":

    print()

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