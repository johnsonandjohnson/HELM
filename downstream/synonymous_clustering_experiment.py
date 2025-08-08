import argparse
from itertools import product

import torch
from dataset import DownstreamDataset
from model_factory import create_model
from helpers import load_config, set_seeds, K_DICT
from utils import clean_sequence_codon
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from create_tree import all_tokens, get_classes
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


codon_list = [
    "GTT", "GTC", "GTA", "GTG",
    "GCT", "GCC", "GCA", "GCG",
    "GAT", "GAC",
    "GAA", "GAG",
    "GGT", "GGC", "GGA", "GGG",
    "TTT", "TTC",
    "TTA", "TTG", "CTT", "CTC", "CTA", "CTG",
    "TCT", "TCC", "TCA", "TCG", "AGT", "AGC",
    "TAT", "TAC",
    "TGT", "TGC",
    "TGG",
    "CCT", "CCC", "CCA", "CCG",
    "CAT", "CAC",
    "CAA", "CAG",
    "CGT", "CGC", "CGA", "CGG", "AGA", "AGG",
    "ATT", "ATC", "ATA",
    "ACT", "ACC", "ACA", "ACG",
    "AAT", "AAC",
    "AAA", "AAG",
    "ATG",
    "TAA", "TAG", "TGA"
]
aa_list = ["V"] * 4 + ["A"] * 4 + ["D"] * 2 + ["E"] * 2 + ["G"] * 4 + ["F"] * 2 + ["L"] * 6 + ["S"] * 6 + ["Y"] * 2 + ["C"] * 2 + ["W"] * 1 + ["P"] * 4 + ["H"] * 2 + ["Q"] * 2 + ["R"] * 6 + ["I"] * 3 + ["T"] * 4 + ["N"] * 2 + ["K"] * 2 + ["start_codon"] + ["end_codon"] * 3 + ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>"]

CODON_TO_AA = {codon: aa for codon, aa in zip(codon_list, aa_list)}

def create_seq_synonymous(seq, codon_to_aa=CODON_TO_AA):
    seq_synonymous = []
    for codon in seq:
        aa = codon_to_aa[codon] if codon in codon_to_aa else '<unk>'
        synonymous_codons = [codon for codon, aa_ in codon_to_aa.items() if aa_ == aa]
        seq_synonymous.append(synonymous_codons)
    return seq_synonymous

def shuffle_all(seq_synonymous):
    new_s = []
    for s in seq_synonymous:
        random.shuffle(s)
        new_s.append(s)
    random.shuffle(new_s)
    return new_s

def generate_seq_synonymous(seq_list):
    list_random_seq = []
    for i, seq in enumerate(product(*seq_list)):
        if i < 100:
            list_random_seq.append(''.join(seq))
        else:
            break
    return list_random_seq

def run_clustering_experiment(model_config_path, model_path, output_path, dataset_config_path, num_sample, seed_index):
    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    seeds = len(dataset_config["path"].keys())
    helm = model_config["helm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed_index >= seeds:
        seed_index = 0
    seed_key = list(dataset_config["path"].keys())[seed_index]
    sequences = pd.read_csv(dataset_config["path"][seed_key]["path_train"])[dataset_config["data_column"]].values
    set_seeds()  # Set seeds for reproducibility

    model, _, _ = create_model(model_config, model_path)
    model.to(device)
    random.shuffle(sequences)
    sequences = sequences[:num_sample]
    selected_sequences = {}
    for idx, sequence in enumerate(sequences):
        try:
            sequence = clean_sequence_codon(sequence)
            sequence = [sequence[i:i+3] for i in range(0, len(sequence) - 2, 3)]
            sequence_synonymous = create_seq_synonymous(sequence)
            sequence_synonymous = shuffle_all(sequence_synonymous)
            product_sequence_synonymous = generate_seq_synonymous(sequence_synonymous)
        except KeyError:
            continue
        for seq in product_sequence_synonymous:
            selected_sequences[seq] = idx

    df = pd.DataFrame(selected_sequences.items(), columns=['Sequence', 'Label'])
    df.to_csv(f'{output_path}/synonymous.csv', index=False)

    tree = None if not helm else get_classes(all_tokens)[0]
    data = DownstreamDataset(f'{output_path}/synonymous.csv', "Sequence", "Label", "codon", tree, k=K_DICT[model_config["tokenizer"]])


    embeddings = []
    labels = []

    for seq, pos, target in tqdm(data):
        seq = seq.unsqueeze(0).to(device)
        pos = pos.unsqueeze(0).to(device)
        target = target.item()
        output = model(seq, position_ids=pos).last_hidden_state.squeeze(0).mean(0).cpu().detach().numpy()
        embeddings.append(output)
        labels.append(target)

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random', perplexity=3).fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig(f"{output_path}/{model_config_path.split('/')[-1].split('.')[0]}_tsne.png")
    k_means = KMeans(n_clusters=100)
    k_means.fit(embeddings)
    # Calculate clustering metrics
    nmi = normalized_mutual_info_score(labels, k_means.labels_)
    mi = mutual_info_score(labels, k_means.labels_)
    # Calculate clustering metrics
    silhouette = silhouette_score(embeddings, k_means.labels_)
    db_index = davies_bouldin_score(embeddings, k_means.labels_)
    ch_index = calinski_harabasz_score(embeddings, k_means.labels_)
    ari = adjusted_rand_score(labels, k_means.labels_)

    # Print the metric scores
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Index: {db_index:.2f}")
    print(f"Calinski-Harabasz Index: {ch_index:.2f}")
    print(f"Adjusted Rand Index: {ari:.2f}")
    print(f"Mutual Information (MI): {mi:.2f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.2f}")

    return silhouette, db_index, ch_index, ari, mi, nmi


parser = argparse.ArgumentParser(description="Run experiments with various configurations")
parser.add_argument("--dataset-config", type=str, help="Path to the dataset configuration file")
parser.add_argument("--seed-index", type=int, help="Seed index to use for the dataset", default=0)
parser.add_argument("--model-config", type=str, help="Path to the model configuration file")
parser.add_argument("--model-path", type=str, help="Path to the model checkpoint")
parser.add_argument("--output-path", type=str, help="Path to save the results")
parser.add_argument("--num-sample", type=int, help="Number of samples to use")

args = parser.parse_args()

if __name__ == "__main__":
    run_clustering_experiment(args.model_config, args.model_path, args.output_path, args.dataset_config, args.num_sample, args.seed_index)