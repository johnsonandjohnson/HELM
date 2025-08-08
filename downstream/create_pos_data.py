import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils import encode_sequence_codon, CODON_TO_ID_3, encode_sequence_aa, AA_TO_ID
from torch.nn import functional as F

class PosData(Dataset):
    def __init__(self,path, data_type, tree) -> None:
        super().__init__()
        CODON_TO_ID = CODON_TO_ID_3
        self.data = pd.read_csv(path)
        self.data_input = self.data["codonized_all"].values
        lens = [len(i) for i in self.data_input]
        self.data_target = self.data[['sp_region_ids','v_region_ids',
       'dj_region_ids', 'c_region_ids']].fillna(0).values
        
        self.data_target = [(int(i[0].split("-")[1]), int(i[1].split("-")[0]), int(i[1].split("-")[1]), int(i[2].split("-")[0]), int(i[2].split("-")[1]), int(i[3].split("-")[0]), int(i[3].split("-")[1])) for i in self.data_target]
        if data_type == "codon":
            self.data_input = [encode_sequence_codon(i, tree=tree) for i in self.data_input]
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
        lens = [len(i) for i in self.data_input]
        self.max_seq_len = max(lens)

    def __len__(self):
        return len(self.data_target)
    
    def __getitem__(self, idx):
        data = self.data_input[idx]
        data = F.pad(torch.tensor(data), (0, self.max_seq_len - len(self.data_input[idx])), 'constant', self.pad).squeeze(0)
        pos_ids = np.arange(len(data))
        target = torch.tensor(self.data_target[idx])
        return data, torch.tensor(pos_ids), target
    


if __name__ == "__main__":
    path = "./oas_codonized_v2_test_split.csv"
    tree = None
    data_type = "codon"
    dataset = PosData(path, data_type, tree)