import argparse
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import pandas as pd
import esm
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from tqdm import tqdm
from scipy.linalg import sqrtm
import re
from kmer_vocab import ID_TO_CODON_3, ID_TO_CODON_1, ID_TO_CODON_6
from model_factory import create_model
from utils import encode_sequence_codon
from create_tree import get_classes, all_tokens
from helpers import load_config, set_seeds, K_DICT
import Levenshtein as lev
from evaluation_models import evaluate

def codon_to_aa(codon):
        return str(Seq(codon.replace("<unk>", "ABC")).translate()).replace("*", "-")


def calculate_internal_diversity(generated_sequences):
    num_seqs = len(generated_sequences)
    if num_seqs < 2:
        return 0.0  # Not enough samples to compare

    dist_matrix = np.zeros((num_seqs, num_seqs))
    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            dist_matrix[i, j] = lev.distance(generated_sequences[i], generated_sequences[j])
            dist_matrix[j, i] = dist_matrix[i, j]  # Symmetric distance matrix

    return np.mean(dist_matrix)

def calculate_precision_recall_f1(distances, threshold=0.5):
        matches = np.sum(distances <= threshold, axis=1)  # Count how many true sequences each generated sequence is close to
        precision = np.mean(matches > 0)  # Proportion of generated sequences that have at least one close match in true sequences
        recall = np.mean(np.any(distances <= threshold, axis=0))  # Proportion of true sequences that are closely matched by any generated sequences
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        return precision, recall, f1_score

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def calculate_gc_content(sequence):
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

def generate(model_config, model_path, dataset_config, seed_index, save_path, num_samples, mode, condition_on=3, temp=1.0, top_k=10, top_p=0.95):
    model_config = load_config(model_config)
    dataset_config = load_config(dataset_config)
    assert model_config["mode"] == "clm", "Generative experiments only supported for CLM models"
    seeds = len(dataset_config["path"].keys())
    helm = model_config["helm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = K_DICT[model_config["tokenizer"]]

    if k == 3:
        ID_TO_CODON = ID_TO_CODON_3
    elif k == 1:
        ID_TO_CODON = ID_TO_CODON_1
    else:
        ID_TO_CODON = ID_TO_CODON_6

    if seed_index >= seeds:
        seed_index = 0
    seed_key = list(dataset_config["path"].keys())[seed_index]
    df = pd.read_csv(dataset_config["path"][seed_key]["path_train"])
    samples = df[dataset_config["data_column"]].values[:num_samples]
    classes = None if not helm else get_classes(all_tokens)[0]

    all_gen_seq = {}

    if mode == "quality":
        assert "dj_region_ids" in df.columns, "dj_region_ids column not found in the dataset"
        j_ids = df["dj_region_ids"].values[:num_samples]
        samples_to_gen = [str(s[:int(j_id.split("-")[0])]) for s, j_id in zip(samples, j_ids)]
        samples_to_gen = [s[:-1] for s in encode_sequence_codon([s for s in samples_to_gen], tree=classes, k=k)]
        lens = [len(s)//k for s in samples]
        all_gen_seq["true"] = samples
    
    elif mode == "property":
        lens = [len(s)//k for s in samples]
        samples_to_gen = [s[:int(len(s) * condition_on)] for s in encode_sequence_codon([s for s in samples], classes, k=k)]
        all_gen_seq["true"] = samples
        all_gen_seq["property"] = df[dataset_config["target_column"]].values[:num_samples]
    

    model, token_ids, _ = create_model(model_config, model_path, generation=True)
    cls_token_id, pad_id, eos_id = token_ids
    model.to(device)
    set_seeds()  # Set seeds for reproducibility

    all_gen_seq["generated"] = []
    for l in tqdm(range(len(samples_to_gen))):
        start_seq = torch.tensor(samples_to_gen[l]).cuda()
        if model_config["model_type"] == "transformer":
            config = {"pad_id":pad_id, "eos_id":eos_id, "cls_id":cls_token_id, "max_length":lens[l], "temperature":temp, "top_k":top_k, "top_p":top_p}
        elif model_config["model_type"] == "mamba":
            config = {"eos_token_id":eos_id, "max_length":lens[l], "temperature":temp, "top_k":top_k, "top_p":top_p}
        gen_seq = model.generate(start_seq.unsqueeze(0), **config)
        gen_seq = gen_seq.cpu().numpy()
        seq = []
        for i in gen_seq[0][1:-1]:
            if not model_config["helm"]:
                seq.append(ID_TO_CODON[i])
            else:
                seq.append(classes[i])
        seq_new = "".join(seq)
        all_gen_seq["generated"].append(seq_new)

    save_path = save_path + f"/generated_sequences_{mode}_data{dataset_config["path"][seed_key]["path_train"].split("/")[-1].split(".")[0]}_temp{temp}.csv"
    df = pd.DataFrame(all_gen_seq)
    df.to_csv(save_path, index=False)
    print(f"Saved generated sequences to {save_path}")
    return df, save_path

def evaluate_quality(df, n_components=4, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds()
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()  # disables dropout for deterministic results
    true = df["true"].values
    gen = df["generated"].values

    true_aa = []
    for idx, item in enumerate(true):
        item = item.replace("T", "U")
        true_aa.append((f"{idx}", codon_to_aa(item)))

    gen_aa = []
    for idx, item in enumerate(gen):
        item = item.replace("T", "U")
        gen_aa.append((f"{idx}", codon_to_aa(item)))

    true_batch = batch_converter(true_aa)
    gen_batch = batch_converter(gen_aa)

    embedding_gen = []
    embedding_true = []

    for item in tqdm(true_batch[2]):
        embedding_true.append(model(item.unsqueeze(0).cuda(), repr_layers=[12], return_contacts=True)["representations"][12].detach().cpu().numpy())

    true_embeddings = np.concatenate(embedding_true, axis=0)

    for item in tqdm(gen_batch[2]):
        embedding_gen.append(model(item.unsqueeze(0).cuda(), repr_layers=[12], return_contacts=True)["representations"][12].detach().cpu().numpy())

    gen_embeddings = np.concatenate(embedding_gen, axis=0)

    max_len = max(true_embeddings.shape[1], gen_embeddings.shape[1])

    def pad_embeddings(embeddings, max_len):
        # Pad or truncate sequences to the max length
        padded = np.zeros((embeddings.shape[0], max_len, embeddings.shape[2]))
        for i in range(embeddings.shape[0]):
            seq_len = embeddings.shape[1]
            if seq_len < max_len:
                padded[i, :seq_len, :] = embeddings[i, :, :]
            else:
                padded[i, :, :] = embeddings[i, :max_len, :]
        return padded

    true_embeddings_padded = pad_embeddings(true_embeddings, max_len)
    gen_embeddings_padded = pad_embeddings(gen_embeddings, max_len)

    # Step 2: Flatten the embeddings to 2D (necessary for PCA)
    true_flattened = true_embeddings_padded.reshape(true_embeddings_padded.shape[0], -1)
    gen_flattened = gen_embeddings_padded.reshape(gen_embeddings_padded.shape[0], -1)

    # Step 3: Calculate the maximum allowable n_components
    n_samples, n_features = true_flattened.shape
    n_components = min(n_samples, n_features, n_components)  # Ensure n_components is valid

    # Step 4: Apply PCA for dimensionality reduction (optional; without it takes ages since the dimensions of embeddings are extremely high)
    pca = PCA(n_components=4)
    true_pca = pca.fit_transform(true_flattened)
    gen_pca = pca.transform(gen_flattened)

    # Step 5: Calculate Gaussian parameters (mean and covariance)
    def calculate_gaussian_params(embeddings_pca):
        mu = np.mean(embeddings_pca, axis=0)
        sigma = np.cov(embeddings_pca, rowvar=False)
        sigma += np.eye(sigma.shape[0]) * 1e-6  # Regularization for numerical stability
        return mu, sigma

    mu_true, sigma_true = calculate_gaussian_params(true_pca)
    mu_gen, sigma_gen = calculate_gaussian_params(gen_pca)

    # Step 6: Calculate Fréchet Biological Distance (FBD)
    fbd = frechet_distance(mu_true, sigma_true, mu_gen, sigma_gen)

    # Step 7: Generate random embeddings for a baseline (for comparison)
    random_embeddings = np.random.randn(*true_embeddings_padded.shape)
    random_flattened = random_embeddings.reshape(random_embeddings.shape[0], -1)
    random_pca = pca.transform(random_flattened)
    mu_random, sigma_random = calculate_gaussian_params(random_pca)

    fbd_random = frechet_distance(mu_true, sigma_true, mu_random, sigma_random)

    print(f"Fréchet Biological Distance (FBD) between True and Gen: {fbd}")
    print(f"Fréchet Biological Distance (FBD) between True and Random: {fbd_random}")

    GC_true = df['true'].apply(calculate_gc_content)
    GC_gen = df['generated'].apply(calculate_gc_content)


    print(f"GC-content of True sequences: {np.mean(GC_true)}%")
    print(f"GC-content of Gen sequences: {np.mean(GC_gen)}%")
   
    def calculate_distance(true, gen, metric='cosine'):
        distances = pairwise_distances(gen, true, metric=metric)
        return distances

    # Calculate distances for each model
    dist_gen = calculate_distance(true_flattened, gen_flattened, metric='cosine')
    
    # Calculate precision, recall, and F1 score for each model
    precision_gen, recall_gen, f1_gen = calculate_precision_recall_f1(dist_gen, threshold)

    print(f"Precision: {precision_gen}, Recall: {recall_gen}, F1 Score: {f1_gen}")
    
    # Example sequences (replace with your actual data)
    gen = df["generated"].to_list()

    # Calculate metrics for each set
    intdiv_gen = calculate_internal_diversity(gen)

    # Print results
    print(f"Generated Internal Diversity (IntDiv) = {intdiv_gen}")

    return {"fbd":fbd, "fbd_random":fbd_random, "gc_true":np.mean(GC_true), "gc_gen":np.mean(GC_gen), "precision":precision_gen, "recall":recall_gen, "f1":f1_gen, "intdiv_gen":intdiv_gen}


def evaluate_property(df, model_type, model_path, head_path):
    true_predicted_labels = evaluate(model_type, model_path, head_path, df["true"].values, df["property"])
    gen_predicted_labels = evaluate(model_type, model_path, head_path, df["generated"].values, df["property"])
    true_labels = df["property"].values
    mse_true = np.power((true_labels - gen_predicted_labels), 2).mean()
    mse_predicted = np.power((true_predicted_labels - gen_predicted_labels), 2).mean()
    data = {"True":true_labels, "Generated":gen_predicted_labels, "Predicted True":true_predicted_labels}
    print(f"MSE True: {mse_true}, MSE Predicted: {mse_predicted}")

    for i in list(data.keys()):
        d = data[i]
        print(f"\nSummary statistics for {i} (0-2 range):")
        print(f"Mean: {np.mean(d):.2f}")
        print(f"Median: {np.median(d):.2f}")
        print(f"Standard Deviation: {np.std(d):.2f}")
        print(f"Min: {np.min(d):.2f}")
        print(f"Max: {np.max(d):.2f}")

    return {"mse_true":mse_true, "mse_predicted":mse_predicted}

def main(model_config, model_path, dataset_config, seed_index, save_path, num_samples, mode, condition_on=1/3, temp=1.0, top_k=10, top_p=0.95, n_components=4, threshold=0.5, eval_model=None, eval_model_path=None, head_path=None):
    df, save_path = generate(model_config, model_path, dataset_config, seed_index, save_path, num_samples, mode, condition_on, temp, top_k, top_p)
    if mode == "quality":
        results = evaluate_quality(df, n_components, threshold)
    elif mode == "property":
        results = evaluate_property(df, eval_model, eval_model_path, head_path)
    return results

parser = argparse.ArgumentParser(description="Run generative experiments with various configurations")
parser.add_argument("--model-config", type=str, help="Path to model configuration file")
parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
parser.add_argument("--dataset-config", type=str, help="Path to dataset configuration file")
parser.add_argument("--seed-index", type=int, help="Seed index to use for the dataset", default=0)
parser.add_argument("--save-path", type=str, help="Path to save generated sequences", default=".")
parser.add_argument("--num-samples", type=int, help="Number of samples to generate", default=2000)
parser.add_argument("--mode", type=str, help="Quality or Property", default="quality", choices=["quality", "property"])
parser.add_argument("--condition-on", type=int, help="Condition on the first n bases", default=3)
parser.add_argument("--temp", type=float, help="Temperature for sampling", default=1.0)
parser.add_argument("--top-k", type=int, help="Top K sampling", default=10)
parser.add_argument("--top-p", type=float, help="Top P sampling", default=0.95)
parser.add_argument("--n-components", type=int, help="Number of components for PCA", default=4)
parser.add_argument("--threshold", type=float, help="Threshold for precision, recall, F1", default=0.5)
parser.add_argument("--eval-model", type=str, help="evaluation model type", default="SpliceBERT", choices=["SpliceBERT", "CodonBert", "OneHotCNN"])
parser.add_argument("--eval-model-path", type=str, help="Path to evaluation model checkpoint", default=None)
parser.add_argument("--head-path", type=str, help="Path to head checkpoint", default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    results = main(args.model_config, args.model_path, args.dataset_config, args.seed_index, args.save_path, args.num_samples, args.mode, args.condition_on, args.temp, args.top_k, args.top_p, args.n_components, args.threshold, args.eval_model, args.eval_model_path, args.head_path)
    print(results)