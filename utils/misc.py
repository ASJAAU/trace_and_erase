import os
import yaml
import glob
import pandas as pd


import torch

### GENERAL IO
def existsfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_valid_files(inputs):
    __EXT__ = (".jpg", ".png", ".bmp")
    images = []
    gts = []
    for input in inputs:
        if os.path.isfile(input) and input.endswith(__EXT__):
            images.append(input)
            gts.append(None)
        elif os.path.isfile(input) and input.endswith('.csv'):
            split = pd.read_csv(input, sep=";")
            images.extend(split["file_name"].to_list())
            gts.extend(split["label"].to_list())
        elif os.path.isdir(input):
            for ext in __EXT__:
                valid_files_in_dir=glob.glob(input + "/*" + ext)
                images.extend(valid_files_in_dir)
                gts.extend([None] * len(valid_files_in_dir))
        else:
            print(f"Invalid input: '{input}'. Skipping")

    return images, gts
         

### CONFIG FILE PARSING
def get_config(path, verbose=False):
    with open (path, 'r') as f:
        cfg = yaml.safe_load(f,)
        #If there is a base config
        if os.path.isfile(cfg["base"]):
            print(f"### LOADING BASE CONFIG PARAMETERS ({cfg['base']}) ####")
            with open (cfg["base"], 'r') as g:
                cfg = update_config(yaml.safe_load(g), cfg)
        else:
            print(f"NO CONFIG BASE DETECTED: Loading '{path}' as is")

    if verbose:
        print(yaml.dump(cfg))
    
    return cfg

def update_config(base_config, updates):
    new_config = base_config
    for key, value in updates.items():
        if type(value) == dict:
            new_config[key] = update_config(new_config[key], value)
        else:
            new_config[key] = value
    return new_config
