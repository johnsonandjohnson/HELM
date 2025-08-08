from functools import partial
import torch
import numpy as np
from torch.nn import functional as F
import pandas as pd
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.text import Perplexity
from dataset import DataAntiBody
from utils import encode_sequence_codon, CODON_TO_ID_3, ID_TO_CODON_3
from lightening_module_hscore import AntibodyLLMLightening
from utils import AA_TO_ID
from create_tree import all_tokens, get_classes
from lightning import Trainer
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from utils import encode_sequence, AA_TO_ID
from torch.nn import functional as F
import os
import argparse
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import yaml

from masking import multi_token_masking
from utils import encode_sequence, AA_TO_ID
from utils import encode_sequence_codon
import torch.nn.functional as F

np.random.seed(42)
noise = np.random.rand(444)

def single_token_masking(ids, pos_ids, mask_ratio, token_to_id, noise, tree=None):
    """Mask a sequence of tokens."""
    assert 0 <= mask_ratio <= 1, "Mask ratio must be between 0 and 1."
    cls_token = token_to_id["<cls>"] if tree is None else tree.index("<cls>")
    eos_token = token_to_id["<eos>"] if tree is None else tree.index("<eos>")
    pad_token = token_to_id["<pad>"] if tree is None else tree.index("<pad>")
    mask_token = token_to_id["<mask>"] if tree is None else tree.index("<mask>")
    ids = np.array(ids)
    special_tokens = (ids == cls_token) | (ids == eos_token) | (ids == pad_token) | (ids == mask_token)
    noise = noise[:len(ids)]
    mask = (noise < np.array([mask_ratio])) & (~special_tokens)
    ids[mask] = mask_token
    return ids, pos_ids

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
                sequences, pos_ids = single_token_masking(sequences, pos_ids, self.masking_ratio, self.token_to_id, noise, self.tree)
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
        
        elif self.mode == "next_token_pred":
            data = self.data[idx]
            data = F.pad(torch.tensor(data), (0, self.max_seq_len - len(self.data[idx])), 'constant', self.pad_token).squeeze(0)
            pos_ids = torch.arange(len(data) - 1)
            return torch.tensor(data[:-1]), torch.tensor(pos_ids), torch.tensor(data[1:])



mode = "next_token_pred"
model = "xe"
path = "./downstream/gpt_clm/last.ckpt"
tree = get_classes(all_tokens)[0]
from hxe import HierarchicalCrossEntropyLoss
from create_tree import all_tokens,  get_weighting, get_classes
weights = get_weighting(all_tokens, "exponential", value=0.2)
classes = get_classes(all_tokens)[0]
if model == "xe":
    criterion = torch.nn.CrossEntropyLoss()
    tokenizer = partial(encode_sequence_codon, tree=None)
    tree = None
elif model == "hxe":
    criterion = HierarchicalCrossEntropyLoss(all_tokens, classes, weights)
    CODON_TO_ID = CODON_TO_ID_3
    tree = classes
    tokenizer = partial(encode_sequence_codon, tree=tree)

from models.transformer import TransformerConfig, Transformer
model_config = TransformerConfig()
model_config.model = "gpt2"
model_config.amp = True
model_config.token_to_id = CODON_TO_ID_3
model_config.bidirectional = True 
model_config.vocab_size = 70
model_config.d_model = 640
model_config.d_intermediate = 2560
model_config.n_layer = 10
model_config.pad_vocab_size_multiple = 1
model_config.distillation = False
model_config.max_position_embeddings = 1024
model_config.pos_embedding = "absolute"
model_config.pos_embedding_apply = "add"
model_config.pos_embedding_dim = None
model_config.distillation_mode = None
model_config.num_heads = 8
model_config.attn_dropout = 0.0
model_config.tree = tree if model == "hxe" else None
model_config.attn_layer_idx = [i for i in range(10)]
model_config.tie_embeddings = False
model = Transformer(config=model_config)
label_metrics = MetricCollection(
    [Accuracy(task="multiclass", num_classes=model.vocab_size)])
logits_metrics = MetricCollection([Perplexity()])
lm = AntibodyLLMLightening.load_from_checkpoint(path, model=model, train_mode=mode, distill_mode=False, distill_type=None, distill_alpha=None, distill_temp=None, teacher_model=None, criterion=criterion, optimizer=None, scheduler=None, label_based_metrics=label_metrics, logits_based_metrics=logits_metrics, token_to_id=CODON_TO_ID_3, tree=tree)
t = Trainer(accelerator="gpu", precision="bf16-mixed")
data = DataAntiBody("./downstream/", "oas_codonized_v2_test_split.csv", tokenizer, CODON_TO_ID_3, 0, ["single"], 0.15, 444, mode, "csv", "codonized_all", tree, 1000)
dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=8)
a = t.predict(lm, dataloaders=dataloader)
output = [b[0] for b in a]
output = np.concatenate(output, axis=0)
target = [b[1] for b in a]
target = np.concatenate(target, axis=0)

import pickle
with open("./downstream/xe.pkl", "wb") as f:
    pickle.dump((output, target), f)