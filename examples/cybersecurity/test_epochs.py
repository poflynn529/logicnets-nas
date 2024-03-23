from architecture import *
import train
import pickle

log_path = "2e-2.pkl"

# Hyper-Parameters
hyper_params = {
    "weight_decay": 0.0,
    "batch_size": 1024,
    "epochs": 100,
    "learning_rate": 2e-2,
    "seed": None,
    "checkpoint": None,
    "log_dir": "training-logs",
    "dataset_file": "unsw_nb15_binarized.npz",
    "cuda": True,
}

# Evaluate accuracy vs. epochs for a standard LogicNets design.

architectures = {
    "nid-m" : nid_m_arch, 
    "nid-m-comp" : nid_m_comp, 
    "nid-s" : nid_s_arch, 
    "nid-s-comp" : nid_s_comp,
    "nid-l" : nid_l_arch,
    "nid-l-comp" : nid_l_comp
}


for arch in [nid_s_arch, nid_m_arch, nid_l_arch]:
    arch.utilisation = arch.compute_utilisation()
    print(arch.compute_loss())


# accuracy = {}

# for key in architectures.keys():
#     arch = Architecture(hyper_params, architectures[key])
#     accuracy[key] = train.main(arch)

# with open(log_path, "wb") as cp_file:
#     pickle.dump(accuracy, cp_file)
