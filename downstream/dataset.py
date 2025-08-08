import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils import encode_sequence_codon, CODON_TO_ID_3, CODON_TO_ID_6, CODON_TO_ID_1, encode_sequence_aa, AA_TO_ID
from torch.nn import functional as F

class DownstreamDataset(Dataset):
    def __init__(self, data_path, data_column, target_column, data_type, tree, k) -> None:
        super().__init__()
        self.path = data_path
        if k == 3:
            CODON_TO_ID = CODON_TO_ID_3
        elif k == 1:
            CODON_TO_ID = CODON_TO_ID_1
        else:
            CODON_TO_ID = CODON_TO_ID_6
        self.data = pd.read_csv(data_path)
        self.data_input = self.data[data_column].values
        self.data_target = self.data[target_column].values
        if data_type == "codon":
            self.data_input = encode_sequence_codon(self.data_input, tree=tree, k=k)
            self.pad = CODON_TO_ID["<pad>"] if tree is None else tree.index("<pad>")
            self.token_to_id = CODON_TO_ID
        elif data_type == "aa":
            if tree:
                self.data_input = [(f"{i}", self.data_input[i]) for i in range(len(self.data_input))]
                self.data_input = tree(self.data_input)[2]
            else:
                self.data_input = [encode_sequence_aa(i) for i in self.data_input]
                self.token_to_id = AA_TO_ID
                self.pad = AA_TO_ID["<pad>"]

        data = []
        target = []
        for idx, i in enumerate(self.data_input):
            if len(i) < 1024:
                data.append(i)
                target.append(self.data_target[idx])

        self.data_input = data
        self.data_target = target
        lens = [len(i) for i in self.data_input]
        self.max_seq_len = max(lens)

    def __len__(self):
        return len(self.data_target)
    
    def __getitem__(self, idx):
        data = self.data_input[idx]
        if len(data) > self.max_seq_len:
            data = torch.tensor(data[:self.max_seq_len])
        else:
            data = F.pad(torch.tensor(data), (0, self.max_seq_len - len(self.data_input[idx])), 'constant', self.pad).squeeze(0)
        pos_ids = np.arange(len(data))
        target = torch.tensor(self.data_target[idx])
        return data, torch.tensor(pos_ids), target
    

