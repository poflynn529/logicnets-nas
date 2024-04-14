#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from logicnets.nn import    generate_truth_tables, \
                            lut_inference, \
                            module_list_to_verilog_module
from logicnets.synthesis import synthesize_and_get_resource_counts
from logicnets.util import proc_postsynth_file

from train import configs, model_config, dataset_config, test
from dataset import get_preqnt_dataset
from models import UnswNb15NeqModel, UnswNb15LutModel, UnswNb15NeqInterLayerModel, UnswNb15NeqInterLayerLutModel

def printlog(text, filename="neq2lut_log.txt"):
    print(text)
    with open(filename, 'a', newline='\n') as file:
        file.write(f"{text}\n")

def generate_verilog(model_cfg, verilog_params, test_params):

    #### Test to see if model loads correctly

    # Fetch the test set
    dataset = {}
    dataset["test"] = get_preqnt_dataset(test_params["dataset_file"], split="test")
    test_loader = DataLoader(dataset["test"], batch_size=test_params['batch_size'], shuffle=False, )

    # Instantiate the PyTorch model
    x, y = dataset["test"][0]
    model_cfg['input_length'] = len(x)
    model_cfg['output_length'] = 1

    if (verilog_params["interlayer"]):
        model = UnswNb15NeqInterLayerModel(model_cfg)
    else:
        model = UnswNb15NeqModel(model_cfg)

    map_location = 'cuda' if verilog_params["cuda"] else 'cpu'

    # Load the model weights
    checkpoint = torch.load(verilog_params["checkpoint_file"], map_location=map_location)
    model.load_state_dict(checkpoint['model_dict'])

    # Test the PyTorch model
    printlog("Running inference on baseline model...")
    model.eval()
    baseline_accuracy = test(model, test_loader, cuda=verilog_params["cuda"])
    printlog(f"Baseline accuracy: {baseline_accuracy}")

    #### Instantiate LUT-based model

    # TODO This is a horribe way of doing this...
    if (verilog_params["interlayer"]):
        lut_model = UnswNb15NeqInterLayerLutModel(model_cfg)
    else:
        lut_model = UnswNb15LutModel(model_cfg)
    lut_model.load_state_dict(checkpoint['model_dict'])

    # Generate the truth tables in the LUT module
    printlog("Converting to NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)
    printlog("Completed truth table generation.")

    # Test the LUT-based model
    printlog("Running inference on LUT-based model...")
    lut_inference(lut_model)
    printlog("Inference complete")
    lut_model.eval()
    printlog("lut eval complete")

    torch.save(lut_model.state_dict(), verilog_params["log_dir"] + "/lut_based_model_no_acc.pth")
    printlog(f"Saved lut_model to {verilog_params['log_dir']}/lut_based_model_no_acc.pth")

    ### Evaluation & Verilog Generation ###

    lut_accuracy = test(lut_model, test_loader, verilog_params["cuda"])
    printlog(f"LUT-Based Model accuracy: {lut_accuracy}")
    modelSave = {   'model_dict': lut_model.state_dict(),
                    'test_accuracy': lut_accuracy}

    printlog(f"Generating verilog in {verilog_params['log_dir']}/verilog")
    module_list_to_verilog_module(lut_model.module_list, "logicnet",verilog_params["log_dir"] + "/verilog", generate_bench=verilog_params["generate_bench"], add_registers=verilog_params["add_registers"])
    printlog(f"Top level entity stored at: {verilog_params['log_dir']}/verilog/logicnet.v ...")

def create_io_file(log_dir):
    io_filename = log_dir + f"io_test.txt"
    with open(io_filename, 'w') as f:
        pass # Create an empty file.
    print(f"Dumping verilog I/O to {io_filename}...")
    return io_filename

def simulate_pre_synth(model_config, verilog_params, test_params):
    print("Running pre-synthesis inference simulation of Verilog-based model...")
    test_loader, input_length = get_test_loader(dataset_file=test_params["dataset_file"], batch_size=test_params["batch_size"])

    model_config['input_length'] = input_length
    model_config['output_length'] = 1
    
    # Test the LUT model
    lut_model = UnswNb15NeqInterLayerLutModel(model_config)
    map_location = 'cuda' if verilog_params["cuda"] else 'cpu'
    checkpoint = torch.load(verilog_params["checkpoint_file"], map_location=map_location)
    lut_model.load_state_dict(checkpoint['model_dict'])
    lut_model.eval()
    baseline_accuracy = test(lut_model, test_loader, cuda=verilog_params["cuda"])
    printlog(f"PyTorch model accuracy: {baseline_accuracy}")

    lut_model.verilog_inference(f"{verilog_params['log_dir']}/verilog", "logicnet.v", logfile=create_io_file(f"{verilog_params['log_dir']}/pre_synth"), unisims_dir="unisims", add_registers=verilog_params["add_registers"])
    verilog_accuracy = test(lut_model, test_loader, cuda=verilog_params["cuda"])
    print("Verilog-Based Model accuracy: %f" % (verilog_accuracy))

def synth(verilog_directory, clk_period_ns=1.0):
    printlog("Running out-of-context synthesis")
    ret = synthesize_and_get_resource_counts(verilog_directory, "logicnet", fpga_part="xcu280-fsvh2892-2L-e", clk_period_ns=clk_period_ns, post_synthesis = 1)

def simulate_post_synth(model_config, verilog_params, test_params):
    print("Running post-synthesis inference simulation of Verilog-based model...")
    test_loader, input_length = get_test_loader(dataset_file=test_params["dataset_file"], batch_size=test_params["batch_size"])
    model_config['input_length'] = input_length
    model_config['output_length'] = 1
    lut_model = UnswNb15NeqInterLayerLutModel(model_config)
    # proc_postsynth_file(log_dir)
    # BTW, I copied the whole unisims directory to my git workspace because my thesis is due next week and I'm not playing around with mounting drives..... Yes I know I'm a gremlin
    lut_model.verilog_inference(f"{verilog_params['log_dir']}/post_synth", f"{model_config['name']}.v", create_io_file(f"{verilog_params['log_dir']}/post_synth"), unisims_dir="unisims", add_registers=verilog_params["add_registers"])
    post_synth_accuracy = test(lut_model, test_loader, cuda=verilog_params["cuda"])
    print("Post-synthesis Verilog-Based Model accuracy: %f" % (post_synth_accuracy))

def get_test_loader(dataset_file, batch_size):
    dataset = {}
    dataset["test"] = get_preqnt_dataset(dataset_file, split="test")
    x, y = dataset["test"][0]

    return DataLoader(dataset["test"], batch_size=batch_size, shuffle=False), len(x)


if __name__ == "__main__":

    model_config_ga_nid_s = {
        "hidden_layers": [298, 79],
        "inter_layer_bitwidth": [1, 2, 1, 2, 2],
        "inter_layer_fanin": [7, 3, 7, 7],
        "name" : "ga_nid_s"
    }

    model_config_ga_nid_sm = {
        "hidden_layers": [425, 191, 128, 128],
        "inter_layer_bitwidth": [1, 1, 1, 2, 1, 1, 2],
        "inter_layer_fanin": [7, 7, 6, 4, 3, 6],
        "name" : "ga_nid_sm"
    }

    model_config_ga_nid_m = {
        "hidden_layers": [425, 193, 128, 54],
        "inter_layer_bitwidth": [1, 1, 1, 2, 1, 1, 2],
        "inter_layer_fanin": [7, 7, 6, 7, 6, 7],
        "name" : "ga_nid_m"
    }

    model_config_ga_nid_l = {
        "hidden_layers": [475, 100, 76, 19, 9, 7],
        "inter_layer_bitwidth": [1, 1, 2, 3, 2, 3, 3, 1, 3],
        "inter_layer_fanin": [6, 4, 7, 6, 6, 4, 6, 6],
        "name" : "ga_nid_l"
    }

    verilog_params = {
    "cuda": True,
    "interlayer": True,
    "log_dir": "ga_nid_l",
    "checkpoint_file": "ga_nid_l/best_accuracy.pth",
    "generate_bench": False,
    "add_registers": True,
    }

    test_params = {
        "batch_size" : 1024,
        "dataset_file" : "unsw_nb15_binarized.npz"
    }

    generate_verilog(model_cfg=model_config_ga_nid_l, verilog_params=verilog_params, test_params=test_params)
    #simulate_pre_synth(model_config=model_config_ga_nid_s, verilog_params=verilog_params, test_params=test_params)
    
