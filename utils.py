import torch
import numpy as np
from kmer_vocab import CODON_TO_ID_1, CODON_TO_ID_3, CODON_TO_ID_6, ID_TO_CODON_1, ID_TO_CODON_3, ID_TO_CODON_6


AA_TO_ID = {'<cls>': 0,
            '<pad>': 1,
            '<eos>': 2,
            '<unk>': 3,
            'L': 4,
            'A': 5,
            'G': 6,
            'V': 7,
            'S': 8,
            'E': 9,
            'R': 10,
            'T': 11,
            'I': 12,
            'D': 13,
            'P': 14,
            'K': 15,
            'Q': 16,
            'N': 17,
            'F': 18,
            'Y': 19,
            'M': 20,
            'H': 21,
            'W': 22,
            'C': 23,
            'X': 24,
            'B': 25,
            'U': 26,
            'Z': 27,
            'O': 28,
            '.': 29,
            '-': 30,
            "start_codon": 31,
            "end_codon": 32,
            '<mask>': 33,
            "<distill_token>": 34,}

ID_TO_AA = {v: k for k, v in AA_TO_ID.items()}


def encode_sequence_aa(sequences):
    """Tokenize a sequence of amino acids and add a cls token at the beginning."""
    encoded_sequences = []
    for sequence in sequences:
        sequence = clean_sequence_aa(sequence)
        tokenized_sequence = [AA_TO_ID[aa] if aa in AA_TO_ID else AA_TO_ID['<unk>'] for aa in sequence]
        tokenized_sequence = [AA_TO_ID['<cls>']] + tokenized_sequence + [AA_TO_ID['<eos>']]
    encoded_sequences.append(tokenized_sequence)
    return encoded_sequences


def decode_sequence_aa(sequence):
    """Decode a sequence of tokens."""
    return "".join([ID_TO_AA[token] if token in ID_TO_AA else "<unk>" for token in sequence])


def clean_sequence_aa(sequence):
    """Remove gaps and convert all residues to upper case."""
    return sequence.replace("-", "").upper()


def encode_sequence_codon(sequences, tree=None, overlap=0, k=3):
    """Tokenize a sequence of amino acids and add a cls token at the beginning."""
    assert k in [1, 3, 6], "k must be 1, 3, or 6."
    assert overlap < k, "Overlap must be smaller than k."
    if tree is not None:
        assert k == 3, "Tree can only be used with k=3."
    
    CODON_TO_ID = CODON_TO_ID_1 if k == 1 else CODON_TO_ID_3 if k == 3 else CODON_TO_ID_6
    encoded_sequences = []
    for sequence in sequences:
        sequence = clean_sequence_codon(sequence)
        sequence = [sequence[i:i+k] for i in range(0, len(sequence) - k + 1, k-overlap)]
        if tree is not None:
            tokenized_sequence = [tree.index(codon) if codon in tree else tree.index('<unk>') for codon in sequence]
            tokenized_sequence = [tree.index("<cls>")] + tokenized_sequence + [tree.index("<eos>")]
        else:
            tokenized_sequence = [CODON_TO_ID[codon] if codon in CODON_TO_ID else CODON_TO_ID['<unk>'] for codon in sequence]
            tokenized_sequence = [CODON_TO_ID["<cls>"]] + tokenized_sequence + [CODON_TO_ID["<eos>"]]

        encoded_sequences.append(tokenized_sequence)
    return encoded_sequences

def decode_sequence_codon(sequence, overlap=0, k=3):
    """Decode a sequence of tokens."""
    assert overlap < k, "Overlap must be smaller than k."
    assert k in [1, 3, 6], "k must be 1, 3, or 6."
    ID_TO_CODON = ID_TO_CODON_1 if k == 1 else ID_TO_CODON_3 if k == 3 else ID_TO_CODON_6
    sequence = [ID_TO_CODON[token] if token in ID_TO_CODON else "<unk>" for token in sequence]
    return "".join(sequence)[::k-overlap]

def clean_sequence_codon(sequence):
    """Remove gaps and convert all residues to upper case."""
    sequence = sequence.strip()
    sequence = sequence.replace("-", "").upper()
    sequence = sequence.replace("U", "T").upper()
    return sequence

def train_tokenizer(data, field, model_type, vocab_size=1000):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, WordPiece, Unigram
    from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
    from tokenizers.normalizers import Lowercase

    if model_type == 'bpe':
        model = BPE(special_tokens=["<cls>", "<eos>", "<unk>", "<mask>", "<pad>", "<distill_token>"])
        trainer = BpeTrainer(special_tokens=["<cls>", "<eos>", "<unk>", "<mask>", "<pad>", "<distill_token>"], vocab_size=vocab_size)
    elif model_type == 'wp':
        model = WordPiece(special_tokens=["<cls>", "<eos>", "<unk>", "<mask>", "<pad>", "<distill_token>"])
        trainer = WordPieceTrainer(special_tokens=["<cls>", "<eos>", "<unk>", "<mask>", "<pad>", "<distill_token>"], vocab_size=vocab_size)
    elif model_type == 'ug':
        model = Unigram()
        trainer = UnigramTrainer(special_tokens=["<cls>", "<eos>", "<unk>", "<mask>", "<pad>", "<distill_token>"], vocab_size=vocab_size)
    else:
        raise ValueError("Model type must be one of 'bpe', 'wp', or 'ug'.")

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = Lowercase()
    def batch_iterator():
        for i in range(0, len(data['train']), 1000):
            cleaned_seq = []
            for seq in data['train'][i:i+1000][field]:
                cleaned_seq.append("<cls>" + clean_sequence_codon(seq) + "<eos>")
            yield cleaned_seq
    
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    return tokenizer

def sample_lengths(len_seq, mask_fraction):
    """
    Sample a length uniformly from 1 to max_L*self.mask_fraction (must be bigger than 1).
    If the length is larger than max_L, return max_L.
    """
    length = np.random.randint(1, max(int(len_seq * mask_fraction), 2))
    return length


if __name__ == "__main__":
    from datasets import Features, Value, ClassLabel
    from datasets import load_dataset

    class_names = ['class_label_1']

    ft = Features({'sequence_heavy': Value('string')})

    dataset = load_dataset("csv", data_files="SRR9179282_paired 2.csv", skiprows=1, features=ft)
    tokenizer = train_tokenizer(dataset, "sequence_heavy", "ug", 1000)
    print("Tokenizer trained successfully.")
    print(tokenizer.encode("<cls>ACGTACGTACGT<eos><pad><pad>").ids)
    print(tokenizer.get_vocab())