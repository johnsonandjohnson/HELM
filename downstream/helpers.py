import random
import os
import json
from create_pos_data import PosData
from dataset import DownstreamDataset
import numpy as np
import torch

def is_potential_path(value):
    """
    Check if the string is likely a path. 
    This function can be adjusted based on your specific use case.
    """
    # Check if it contains path separators or starts with '.', indicating a relative path
    return  ('/' in value or '\\' in value or value.startswith('.')) if isinstance(value, str) else False

def adjust_paths(config, base_dir):
    """
    Recursively adjust relative paths in the config dictionary.
    """
    if isinstance(config, dict):
        # Traverse through dictionary items
        for key, value in config.items():
            config[key] = adjust_paths(value, base_dir)
    elif isinstance(config, list):
        # Traverse through list items
        config = [adjust_paths(item, base_dir) for item in config]
    elif is_potential_path(config) and not os.path.isabs(config):
        # Adjust relative path strings
        return os.path.normpath(os.path.join(base_dir, config))
    
    return config

def load_config(config_path):
    # Get the directory of the config file
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Adjust paths in the config recursively
    return adjust_paths(config, config_dir)

def get_numpy_value(numpy_obj):
    # Check if it's a NumPy scalar
    if np.isscalar(numpy_obj):
        return numpy_obj.item()

    # Check if it's a NumPy array with a single element
    elif isinstance(numpy_obj, np.ndarray) and numpy_obj.size == 1:
        return numpy_obj.item()

    # If it's not scalar or a single-element array, return as is
    return numpy_obj

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


K_DICT = {
    "nuc": 1,
    "3mer": 3,
    "6mer": 6
}

def prepare_data(path_train, path_val, path_test, tokenizer, data_column, target_column, helm):
    k = K_DICT[tokenizer]
    if "oas" in path_train:
        if helm:
            from create_tree import all_tokens, get_classes
            classes = get_classes(all_tokens)[0]
            data_train = PosData(path_train, "codon", classes)
            data_val = PosData(path_val, "codon", classes)
            data_test = PosData(path_test, "codon", classes)
        else:
            data_train = PosData(path_train, "codon", None)
            data_val = PosData(path_val, "codon", None)
            data_test = PosData(path_test,"codon", None)
    else:
        data_train = DownstreamDataset(path_train, data_column, target_column, "codon", None, k)
        data_val = DownstreamDataset(path_val, data_column, target_column, "codon", None, k)
        data_test = DownstreamDataset(path_test, data_column, target_column, "codon", None, k)
        if helm:
            from create_tree import all_tokens, get_classes
            classes = get_classes(all_tokens)[0]
            data_train = DownstreamDataset(path_train, data_column, target_column, "codon", classes, k)
            data_val = DownstreamDataset(path_val, data_column, target_column, "codon", classes, k)
            data_test = DownstreamDataset(path_test, data_column, target_column, "codon", classes, k)

    return data_train, data_val, data_test