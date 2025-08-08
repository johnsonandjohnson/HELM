# Downstream

This repository contains scripts for running all downstream model fine-tuning and generative experiments utilizing various configurations for model architecture, datasets, and fine-tuning methods.

## Directory Structure

```plaintext
downstream/
│
├── data/                     # Data folder (see README for more information on data preparation)
│
├── model_configs/            # Model configurations folder (see README for model config creation)
│
├── models/                   # Pre-trained models and checkpoints
│
├── create_pos_data.py        # Script to create position data for annotation experiment
├── create_tree.py            # Script to create hierarchical tree structure
├── dataset.py                # Dataset loading and preparation module
├── downstream_model.py       # Core model training module for downstream tasks
├── evaluation_metrics.py     # Evaluation metrics calculation module
├── evaluation_models.py      # Evaluation models for downstream tasks
├── helpers.py                # Helper functions for various utilities
├── hxe.py                    # Hyperbolic embeddings script
├── kmer_vocab.py             # Script for k-mer vocabulary creation
├── lightening_module.py      # PyTorch Lightning module for training
├── model_factory.py          # Model factory to create models from configurations
├── README.md                 # Main README file
├── run_downstream.py         # Script to run downstream model fine-tuning
├── run_generative_experiments.py  # Script to run generative experiments
├── run.sh                    # Bash script to run experiments in parallel
├── synonymous_clustering_experiment.py  # Synonymous clustering experiment script
└── utils.py                  # Utility functions for the repository
```

## Overview

This repository supports a range of downstream evaluation tasks, from model fine-tuning to generating and clustering synonymous sequences. It includes scripts for supervised and unsupervised tasks, with flexibility for different models, datasets, and fine-tuning methods.

## Data and Configs

[Data README](./data/README.md): Detailed data descriptions, available datasets, and preparation guides.

[Model Configs README](./model_configs/README.md): Instructions on creating and modifying model configurations, with a list of available models.

## Downstream Performance Evaluation

### 1. `run_downstream.py`

This Python script handles the downstream fine-tuning of models using different configurations, supporting fine-tuning options such as LoRA and full fine-tuning.

**Key Features:**

- Load datasets for regression or classification tasks.
- Initialize models and configure them with options like LoRA or full fine-tuning.
- Set up optimizers and schedulers.
- Train models with specified datasets and hyperparameters.
- Log performance metrics during training.

#### Usage

Run the script with the following command-line arguments:

```bash
python run_downstream.py --gpu-id <GPU_ID> \
                         --batch-size <BATCH_SIZE> \
                         --learning-rate-head <LR_HEAD> \
                         --learning-rate-backbone <LR_BACKBONE> \
                         --model-config <MODEL_CONFIG_PATH> \
                         --model-path <MODEL_CHECKPOINT> \
                         --dataset-config <DATASET_CONFIG_PATH> \
                         --seed-index <SEED_INDEX> \
                         --finetune <FINETUNE_METHOD> \
                         --output-path <OUTPUT_PATH>
```

#### Arguments

- `--gpu-id`: GPU ID for training (e.g., `0`, `1`), supporting parallel execution in `run.sh`.
- `--batch-size`: Batch size for training (e.g., 8, 16, 32, 64).
- `--learning-rate-head`: Learning rate for the model head.
- `--learning-rate-backbone`: Learning rate for the model backbone.
- `--model-config`: Path to the model configuration file.
- `--model-path`: Path to the model checkpoint.
- `--dataset-config`: Path to the dataset configuration file.
- `--seed-index`: Seed index for dataset selection.
- `--finetune`: Finetuning method (`none`, `full`, `lora`).
- `--output-path`: Path to save results.

#### Example

```bash
python run_downstream.py --gpu-id 0 --batch-size 16 --learning-rate-head 0.0003 --model-config /path/to/model/config.json --model-path /path/to/checkpoint.ckpt --dataset-config /path/to/dataset/config.json --seed-index 0 --finetune lora --output-path /path/to/output.json
```

### 2. `run.sh`

This bash script automates a series of experiments based on predefined configurations, distributing jobs across available GPUs for parallel execution.

#### Configuration

The script includes predefined arrays for:

- `config_batch_size`: Batch sizes to experiment with.
- `config_learning_rate`: Learning rates to try.
- `config_model`: List of model configurations and checkpoints.
- `config_dataset`: List of dataset configuration paths.
- `config_dataset_seed`: Seeds to run for each dataset.
- `config_finetune_method`: Fine-tuning methods (`none`, `lora`, etc.).
- `output_path`: Output path to save results.
- `gpu_ids`: Available GPU IDs for running experiments (e.g., `gpu_ids=(0 1 2)` for parallel jobs on GPUs 0, 1, and 2).

#### Usage

Run the script as follows:

```bash
bash run.sh
```

#### Customization

To add or modify configurations, update the respective arrays in the script:

```bash
config_batch_size=(8 16 32 64)
config_learning_rate=(3e-4 1e-4 1e-5)
config_model=(
    "/path/to/config1.json /path/to/checkpoint1.ckpt"
    "/path/to/config2.json /path/to/checkpoint2.ckpt"
)
config_dataset=(/path/to/dataset/config.json)
```

## Generative Performance Evaluation

This section details how to run generative experiments to evaluate model performance in generating biologically relevant sequences.

### `run_generative_experiments.py`

This script generates sequences using specified language models and evaluates their quality and properties, including metrics like Fréchet Biological Distance (FBD), internal diversity, and GC-content.

**Main Functions:**

- **`generate()`**: Generates sequences using the model.
- **`evaluate_quality()`**: Computes metrics like FBD, precision, recall, F1-score, and internal diversity.
- **`evaluate_property()`**: Evaluates properties of generated sequences.
- **`main()`**: Executes generation and evaluation based on parameters.

#### Arguments

- `--model-config`: Path to the model configuration file.
- `--model-path`: Path to the model checkpoint.
- `--dataset-config`: Path to the dataset configuration file.
- `--seed-index`: Seed index for dataset.
- `--save-path`: Path to save generated sequences.
- `--num-samples`: Number of sequences to generate.
- `--mode`: Generation mode ('quality' or 'property').
- `--condition-on`: Number of bases to condition on.
- `--temp`: Sampling temperature.
- `--top-k`: Top K sampling.
- `--top-p`: Top P sampling.
- `--n-components`: PCA components during evaluation.
- `--threshold`: Distance threshold for precision, recall, and F1-score.
- `--eval-model`: Evaluation model type.
- `--eval-model-path`: Path to evaluation model checkpoint.
- `--head-path`: Path to the evaluation model head checkpoint.

#### Example Usage

For quality evaluation:

```bash
python run_generative_experiments.py --model-config $MODEL_CONFIG --model-path $MODEL_CHECKPOINT --dataset-config $DATA_CONFIG --save-path $OUTPUT_PATH --mode quality
```

For property evaluation:

```bash
python run_generative_experiments.py --model-config $MODEL_CONFIG --model-path $MODEL_CHECKPOINT --dataset-config $DATA_CONFIG --save-path $OUTPUT_PATH --mode property --eval-model-path $EVAL_MODEL_PATH --head-path $HEAD_PATH
```

## Synonymous Clustering Experiment

### `synonymous_clustering_experiment.py`

This script evaluates clustering of synonymous codon sequences, providing metrics like Silhouette Score and t-SNE visualizations.

**Main Functions:**

- **`create_seq_synonymous()`**: Maps input sequences to synonymous codons.
- **`shuffle_all()`**: Randomly shuffles synonymous codon lists.
- **`generate_seq_synonymous()`**: Generates synonymous sequences.
- **`run_clustering_experiment()`**: Runs the clustering experiment, including embedding extraction, t-SNE visualization, and clustering metrics evaluation.

#### Arguments
The script supports several command-line arguments:
- `--dataset-config`: Path to the dataset configuration file.
- `--seed-index`: Seed index to use for the dataset (default: 0).
- `--model-config`: Path to the model configuration file.
- `--model-path`: Path to the model checkpoint.
- `--output-path`: Path to save the results, including t-SNE plots and clustering metrics.
- `--num-sample`: Number of samples to use for the experiment.


#### Example Usage

```bash
python synonymous_clustering_experiment.py --dataset-config configs/dataset_config.json --model-config configs/model_config.json --model-path checkpoints/model.pt --output-path results/ --num-sample 1000 --seed-index 0
```
