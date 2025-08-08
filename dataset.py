import argparse
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import yaml

from masking import single_token_masking, multi_token_masking
from utils import encode_sequence_aa, AA_TO_ID, encode_sequence_codon
import torch.nn.functional as F


class DataAntiBody(Dataset):
    def __init__(self, path, file, tokenizer, token_to_id, position_ids, masking_strategy, masking_ratio, max_seq_len, mode, extension="txt", column_name=None, tree=None, num_samples=-1):
        p = os.path.join(path, file)
        if extension == "csv":
            assert column_name is not None, "Column name must be provided for csv files."
            df = pd.read_csv(p, skiprows=0)
            if num_samples == -1:
                self.data = df[column_name].values
            else:
                self.data = df[column_name].values[0:num_samples]

        if extension == "txt":
            with open(p, "r") as f:
                if num_samples == -1:
                    self.data = f.readlines()
                else:
                    self.data = f.readlines()[0:num_samples]
        
        self.data = tokenizer(self.data)
        self.position_ids = position_ids
        self.masking_strategy = masking_strategy
        self.masking_ratio = masking_ratio
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.token_to_id = token_to_id
        self.pad_token = self.token_to_id["<pad>"] if tree is None else tree.index("<pad>")
        self.lens = [len(d) for d in self.data]
        self.tree = tree
        print(max(self.lens))
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == "mlm":
            sequences = self.data[idx].copy()
            pos_ids = np.arange(len(self.data[idx]))
            if "single" in self.masking_strategy:
                sequences, pos_ids = single_token_masking(sequences, pos_ids, self.masking_ratio, self.token_to_id, self.tree)
            if "film" in self.masking_strategy:
                sequences, pos_ids = multi_token_masking(sequences, pos_ids, 1, mask_ratio=0.15, token_to_id=self.token_to_id)
            
            seq_size = sequences.shape[0]
            
            if seq_size >= self.max_seq_len:
                sequences = sequences[:self.max_seq_len]
                pos_ids = pos_ids[:self.max_seq_len]
                target = self.data[idx][:self.max_seq_len]

            else:
                sequences = F.pad(torch.tensor(sequences), (0, self.max_seq_len - len(self.data[idx])), 'constant',
                                self.pad_token).squeeze(0)
                pos_ids = torch.concatenate([torch.tensor(pos_ids), torch.arange(seq_size, sequences.shape[0])])
                target = F.pad(torch.tensor(self.data[idx]), (0, self.max_seq_len - len(self.data[idx])), 'constant',
                            self.pad_token).squeeze(0)
            
            return torch.tensor(sequences), torch.tensor(pos_ids), torch.tensor(target)
        
        elif self.mode == "clm":
            data = self.data[idx]
            data = F.pad(torch.tensor(data), (0, self.max_seq_len - len(self.data[idx])), 'constant', self.pad_token).squeeze(0)
            pos_ids = torch.arange(len(data) - 1)
            return torch.tensor(data[:-1]), torch.tensor(pos_ids), torch.tensor(data[1:])

