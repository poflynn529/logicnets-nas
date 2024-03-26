import pandas as pd

def make_seed(hidden_layers, inter_layer_bitwidth, inter_layer_fanin, proxy_accuracy=None, training_epochs=None, learning_rate=None):
    seed_population = pd.DataFrame(columns=['gen', 'hash', 'hidden_layers', 'inter_layer_bitwidth', 'inter_layer_fanin', 'proxy_utilisation', 'proxy_accuracy', 'fitness', 'evaluated', 'training_epochs', 'learning_rate'])
    param_dict = {
        "gen"                  : 0,
        "hidden_layers"        : hidden_layers,
        "inter_layer_bitwidth" : inter_layer_bitwidth, 
        "inter_layer_fanin"    : inter_layer_fanin,
        "evaluated"            : False
    }

    if proxy_accuracy != None and training_epochs != None and learning_rate != None:
        param_dict["proxy_accuracy"] = proxy_accuracy
        param_dict["training_epochs"] = training_epochs
        param_dict["learning_rate"] = learning_rate
        param_dict["evaluated"] = True

    return pd.concat([seed_population, pd.DataFrame([param_dict], columns=seed_population.columns)], ignore_index=True)

initial_seed = make_seed(hidden_layers=[593, 256, 128, 128], inter_layer_bitwidth=[1, 2, 2, 2, 2, 2, 2], inter_layer_fanin=[7, 7, 7, 7, 7, 7]) # LogicNets NID-M

initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[288, 144, 132, 115], inter_layer_bitwidth=[1, 1, 1, 1, 2, 1, 2], inter_layer_fanin=[4, 7, 7, 7, 4, 4])]) # Known good architecture
initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[464, 238, 197, 22], inter_layer_bitwidth=[1, 2, 2, 1, 2, 1, 2], inter_layer_fanin=[4, 3, 5, 5, 4, 4])]) # Known good architecture
initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[33, 16, 6, 5], inter_layer_bitwidth=[1, 2, 2, 2, 2, 2, 2], inter_layer_fanin=[7, 7, 7, 7, 7, 7])]) # Known good architecture
initial_seed = pd.concat([initial_seed, make_seed(hidden_layers=[97, 39, 14, 7], inter_layer_bitwidth=[1, 2, 2, 2, 2, 2, 2], inter_layer_fanin=[7, 7, 7, 7, 7, 7])]) # Known good architecture

initial_seed.reset_index(drop=True, inplace=True)

initial_seed.to_pickle("nas_pickle/seed0.pkl")

print(initial_seed)