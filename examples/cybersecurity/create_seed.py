import pandas as pd
import yaml
from architecture import *

with open('hyper_params.yaml', 'r') as file:
    hyper_params = yaml.safe_load(file)

def make_seed(hidden_layers, inter_layer_bitwidth, inter_layer_fanin, proxy_accuracy=None, h_params=None):

    seed_population = pd.DataFrame(columns=['gen', 'hash', 'hidden_layers', 'inter_layer_bitwidth', 'inter_layer_fanin', 'proxy_utilisation', 'proxy_accuracy', 'fitness', 'evaluated', 'training_epochs', 'learning_rate'])
    
    if proxy_accuracy == None:
        param_dict = {
            "gen"                  : 0,
            "hidden_layers"        : hidden_layers,
            "inter_layer_bitwidth" : inter_layer_bitwidth, 
            "inter_layer_fanin"    : inter_layer_fanin,
            "evaluated"            : False
        }
    elif proxy_accuracy != None and h_params != None:
        param_dict = {
            "gen"                  : 0,
            "hash"                 : Architecture.compute_hash(hidden_layers, inter_layer_bitwidth, inter_layer_fanin),
            "hidden_layers"        : hidden_layers,
            "inter_layer_bitwidth" : inter_layer_bitwidth, 
            "inter_layer_fanin"    : inter_layer_fanin,
            "proxy_utilisation"    : Architecture.compute_utilisation(hidden_layers, inter_layer_bitwidth, inter_layer_fanin),
            "proxy_accuracy"       : proxy_accuracy,
            "evaluated"            : True,
            "training_epochs"      : h_params["epochs"],
            "learning_rate"        : h_params["learning_rate"],
        }
        param_dict["fitness"] = Architecture.compute_loss(proxy_accuracy, param_dict["proxy_utilisation"], hyper_params=h_params)
    else:
        raise(ValueError("hyper_params cannot be None if proxy accuracy has been defined!"))

    return pd.concat([seed_population, pd.DataFrame([param_dict], columns=seed_population.columns)], ignore_index=True)

initial_seed = make_seed(hidden_layers=[593, 193, 128, 128], inter_layer_bitwidth=[1, 2, 1, 2, 1, 2, 2], inter_layer_fanin=[7, 7, 6, 7, 7, 7], proxy_accuracy=0.859, h_params=hyper_params) # Known good architecture
initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[288, 144, 132, 115], inter_layer_bitwidth=[1, 1, 1, 1, 2, 1, 2], inter_layer_fanin=[4, 7, 7, 7, 4, 4], proxy_accuracy=0.801, h_params=hyper_params)]) # Known good architecture
initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[464, 238, 197, 22], inter_layer_bitwidth=[1, 2, 2, 1, 2, 1, 2], inter_layer_fanin=[4, 3, 5, 5, 4, 4], proxy_accuracy=0.777, h_params=hyper_params)]) # Known good architecture
initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[33, 16, 6, 5], inter_layer_bitwidth=[1, 2, 2, 2, 2, 2, 2], inter_layer_fanin=[7, 7, 7, 7, 7, 7], proxy_accuracy=0.762, h_params=hyper_params)]) # Known good architecture
initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[97, 39, 14, 7], inter_layer_bitwidth=[1, 2, 2, 2, 2, 2, 2], inter_layer_fanin=[7, 7, 7, 7, 7, 7], proxy_accuracy=0.822, h_params=hyper_params)]) # Known good architecture

initial_seed.reset_index(drop=True, inplace=True)

initial_seed.to_pickle("nas_pickle/seed0.pkl")

print(initial_seed)