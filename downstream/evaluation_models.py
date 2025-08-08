
import torch
from downstream_model import TextCNN
from models.hf_models.bert import BertForSequenceClassification

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import *
from tokenizers.processors import BertProcessing

from transformers import PreTrainedTokenizerFast, BertForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def mytok(seq, kmer_len, s):
        seq = seq.upper().replace("T", "U")
        kmer_list = []
        for j in range(0, (len(seq)-kmer_len)+1, s):
            kmer_list.append(seq[j:j+kmer_len])
        return kmer_list

def build_dataset(seqs, ys, max_length=1024):
    total = len(seqs)
    processed_seqs = []
    processed_ys = []
    for seq, y in zip(seqs, ys):
        lst_tok = mytok(seq, 3, 3)
        if lst_tok:
            if len(lst_tok) > max_length - 2:
                skipped += 1
                print("Skip one sequence with length", len(lst_tok), \
                    "codons. Skipped %d seqs from total %d seqs." % (skipped, total))
                continue
            processed_seqs.append(" ".join(lst_tok))
            processed_ys.append(float(y))
    ds_train = Dataset.from_list([{"seq": seq, "label": y}])

    return ds_train

def one_hot_encode(sequence, len_seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '<': 4, 'u': 4, 'n': 4, 'k': 4, '>': 4}
    one_hot = np.zeros((len_seq, 5), dtype=int)
    for i, nucleotide in enumerate(sequence):
        one_hot[i, mapping[nucleotide]] = 1
    return one_hot 

def evaluate(model_type, model_path, head_path, sequences, target):
    assert model_type in ["SpliceBERT", "OneHotCNN", "CodonBert"], "Model type not supported"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_type == "CodonBert":
        # default Model hyperparameters (do not change)
        max_length = 1024

        lst_ele = list('AUGC')
        lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        for a1 in lst_ele:
            for a2 in lst_ele:
                for a3 in lst_ele:
                    lst_voc.extend([f'{a1}{a2}{a3}'])

        dic_voc = dict(zip(lst_voc, range(len(lst_voc))))

        tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
        tokenizer.add_special_tokens(['[PAD]','[CLS]', '[UNK]', '[SEP]','[MASK]'])
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = BertProcessing(
            ("[SEP]", dic_voc['[SEP]']),
            ("[CLS]", dic_voc['[CLS]']),
        )

        bert_tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                                    do_lower_case=False,
                                                    clean_text=False,
                                                    tokenize_chinese_chars=False,
                                                    strip_accents=False,
                                                    unk_token='[UNK]',
                                                    sep_token='[SEP]',
                                                    pad_token='[PAD]',
                                                    cls_token='[CLS]',
                                                    mask_token='[MASK]')

        def encode_string(data):
            return bert_tokenizer_fast(data['seq'],
                                    truncation=True,
                                    padding="max_length",
                                    max_length=max_length,
                                    return_special_tokens_mask=True)


        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=1).bert
        model.to(device)
        head = TextCNN(768, 640, 1, True, 100)
        head.load_state_dict(torch.load(head_path))
        head.eval()
        head.to(device)

        ds_train = build_dataset(sequences, target, max_length=max_length)
        train_loader = DataLoader(ds_train.map(encode_string, batched=True), batch_size=64, shuffle=False)
        sequences_predicted_labels = []
        for ip in tqdm(train_loader):
            inputs = torch.stack(ip["input_ids"], dim=0).T.to(device)
            attention_mask = torch.stack(ip["attention_mask"]).T.to(device)
            token_type_ids = torch.stack(ip["token_type_ids"]).T.to(device)
            with torch.no_grad():
                outputs = model(input_ids=inputs, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
                outputs = (head(outputs) > 0.5).float()
            sequences_predicted_labels.append(outputs.cpu().detach().numpy())

        sequences_predicted_labels = np.concatenate(sequences_predicted_labels)

    elif model_type == "SpliceBERT":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)# load model
        head = TextCNN(512, 640, 1, True, 100)
        head.load_state_dict( torch.load(head_path))
        head.to(device)
        model.eval()
        head.eval()
        sequences_predicted_labels = []
        for seq in tqdm(sequences):
            seq = ' '.join(list(seq.upper().replace("U", "T"))) # U -> T and add whitespace
            input_ids = tokenizer.encode(seq) # N -> 5, A -> 6, C -> 7, G -> 8, T(U) -> 9. NOTE: a [CLS] and a [SEP] token will be added to the start and the end of seq
            input_ids = torch.as_tensor(input_ids) # convert python list to Tensor
            input_ids = input_ids.unsqueeze(0).cuda() # add batch dimension, shape: (batch_size, sequence_length)
            with torch.no_grad():
                last_hidden_state = model(input_ids).last_hidden_state
                output = head(last_hidden_state)
            sequences_predicted_labels.append(output.item())
    
    elif model_type == "OneHotCNN":
        head = TextCNN(5, 640, 1, True)
        head.load_state_dict(torch.load(head_path))
        head.to(device)
        head.eval()
        sequences_encoded = [one_hot_encode(seq, 902) for seq in sequences]
        sequences_predicted_labels = []
        for seq in tqdm(sequences_encoded):
            input_ids = torch.tensor(seq).unsqueeze(0).to(device)
            with torch.no_grad():
                output = head(input_ids)
            sequences_predicted_labels.append(output.item())

    else:
        raise ValueError("Model type not supported")
    
    return np.array(sequences_predicted_labels)