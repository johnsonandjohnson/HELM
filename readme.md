# HELM: Hierarchical Encoding for mRNA Language Modeling

This repository provides the implementation and experimental setup for the paper:  
**[HELM: Hierarchical Encoding for mRNA Language Modeling](https://arxiv.org/abs/2410.12459)**

This repository contains a PyTorch Lightning pipeline for training Antibody Large Language Models (LLMs) focused on mRNA sequence modeling. The `train.py` script allows for training with various transformer-based architectures such as GPT-2, BERT, Mamba, and Hyena, while utilizing tokenization strategies specific to mRNA sequences.

## Requirements

Install the necessary packages using:

```bash
pip install -r requirements.txt
```

Ensure a compatible GPU setup along with the latest versions of CUDA and cuDNN for optimal performance with PyTorch.

## Usage

You can run the `train.py` script with different command-line arguments, as outlined below. Alternatively, a configuration YAML file can be provided via the `--config` argument.

### Basic Command

```bash
python train.py --data_dir /path/to/data --train_file train.csv --model_name gpt2 --batch_size 64 --epochs 300
```

### Key Arguments

#### Loss Parameters:
- `--loss`: Specifies the loss function to be used for training. Options include:
  - **XE**: Standard Cross Entropy, which treats each codon as an independent class, focusing on direct accuracy without considering hierarchical relationships.
  - **HXE**: Hierarchical Cross Entropy, which leverages the hierarchy of codons to apply varying penalties depending on the level of the misclassification. This is useful for learning broader codon relationships based on functional grouping.
  - **aa_loss_add** and **aa_loss_only**: Specialized amino acid loss functions that incorporate amino acid grouping.
  - **soft_labels**: Uses soft labels to provide probabilistic targets that allow the model to make “better” mistakes by assigning higher probabilities to similar codons within the same functional group.

- `--hxe-alpha`: Controls the rate at which penalties decrease across hierarchical levels when using HXE. Higher values cause more significant penalties for top-level misclassifications, encouraging the model to focus on accurately classifying codon groups rather than only individual codons. Suggested values: `0.2`, `0.4`, `0.6`.

- `--soft-labels-beta`: Defines the degree of smoothing for soft labels. Higher beta values result in softer distributions, making the model more lenient and assigning some probability to similar codon classes. This helps in capturing the hierarchical structure and encouraging the model to make mistakes within functionally similar classes. Suggested values typically range from `5` to `30`.
- `--hierarchy`: Specifies the type of hierarchy used for organizing codons, applicable only when using `soft_labels` or `HXE` as the loss function. This influences how relationships between codons are considered during training. Options include:
  - **true**: Uses a biologically accurate hierarchy, where codons are grouped based on their amino acid and functional relationships, such as start codons, amino acids, and stop codons.
  - **random**: Creates a random hierarchy that does not correspond to true biological significance. This can be useful for exploring how the model generalizes when codons are grouped without biological meaning, potentially aiding in understanding the model's behavior without predefined structures.



#### Data Parameters:
- `--data_dir`: Directory containing data files.
- `--train_file`, `--val_file`, `--test_file`: Filenames for training, validation, and test datasets.
- `--file_extension`: File extension for data files (default: `txt`).
- `--column_name`: Column name with sequence data in CSV files.
- `--max_seq_len`: Maximum sequence length for the model input (default: `256`).
- `--masking_strategy`: Masking strategy, choose from `single`, `double`, or `span`.
- `--masking_ratio`: Ratio of tokens to mask (default: `0.15`).
- `--data-type`: Data type (nucleotide or amino acid) for training (`nuc` or `aa`).
- `--tokenizer`: Tokenization strategy (`3mer`, `nuc`, `6mer`, `bpe`, `wp`, `ug`).
- `--overlap`: Overlap between tokens (default: `0`).
- `--position_ids`: Use position IDs (default: `False`).
- `--mode`: Training mode (`mlm` or `clm`).
- `--batch_size`: Training batch size (default: `64`).
- `--num_samples`: Number of samples for training (default: `-1` for all samples).

#### Model Parameters:
- `--model_name`: Model type (`gpt2`, `bert`, `mamba`, `hyena`).
- `--bidirectional`: Use bidirectional model (for BERT-style models, default: `False`).
- `--ssm_layer`: Select SSM layer type (`Mamba`, default).
- `--vocab_size`: Vocabulary size for tokenization (default: `70`). (Redundant)
- `--num_layers`: Number of layers in the model (default: `4`).
- `--num_heads`: Number of attention heads (default: `4`).
- `--pad_vocab_size_multiple`: Vocabulary padding multiple (default: `1`).
- `--pos_embedding`: Type of position embedding to use.
- `--pos_embedding_apply`: How to apply position embedding (`add`, default).
- `--pos_embedding_dim`: Dimension of the position embedding.
- `--tie_embeddings`: Tie embeddings between input and output (default: `False`).
- `--hidden_size`: Size of hidden layers (default: `256`).
- `--intermediate_size`: Intermediate layer size (default: `1024`).
- `--dropout`: Dropout rate (default: `0.0`).
- `--attn_dropout`: Attention dropout rate (default: `0.0`).
- `--activation`: Activation function (`gelu`, default).

#### Optimizer Parameters:
- `--loss`: Loss function (`XE`, `HXE`, `aa_loss_add`, `aa_loss_only`, `soft_labels`).
- `--hxe-alpha`: Alpha value for Hierarchical Cross Entropy (default: `0.2`).
- `--soft-labels-beta`: Beta value for soft labels (default: `15`).
- `--hierarchy`: Type of hierarchy (`random` for non-biological).
- `--opt`: Optimizer type (`adamw`, default).
- `--opt-eps`: Optimizer epsilon value.
- `--opt-betas`: Optimizer betas.
- `--momentum`: Momentum (default: `0.9`).
- `--weight-decay`: Weight decay (default: `2e-5`).
- `--clip-grad`: Clip gradient norm.
- `--clip-mode`: Clip mode (`norm`, `value`, `agc`).
- `--layer-decay`: Layer-wise learning rate decay.
- `--grad-accum-steps`: Gradient accumulation steps (default: `1`).

#### Learning Rate Schedule Parameters:
- `--early-stop`: Enable early stopping (default: `False`).
- `--sched`: Scheduler type (`cosine`, default).
- `--sched-on-updates`: Apply scheduler step on update.
- `--lr`: Learning rate.
- `--lr-base`: Base learning rate.
- `--lr-base-size`: Batch size divisor for learning rate (default: `256`).
- `--lr-noise`: Learning rate noise percentages.
- `--lr-noise-pct`: Noise limit percentage (default: `0.67`).
- `--lr-noise-std`: Noise standard deviation (default: `1.0`).
- `--lr-cycle-mul`: Learning rate cycle multiplier (default: `1.0`).
- `--lr-cycle-decay`: Cycle decay amount (default: `0.5`).
- `--lr-cycle-limit`: Cycle limit (default: `1`).
- `--lr-k-decay`: Learning rate k-decay factor (default: `1.0`).
- `--warmup-lr`: Warmup learning rate (default: `1e-8`).
- `--min-lr`: Minimum learning rate.
- `--epochs`: Number of epochs (default: `300`).
- `--epoch-repeats`: Epoch repeat multiplier.
- `--start-epoch`: Manual start epoch for restarts.
- `--decay-milestones`: Epoch milestones for learning rate decay.
- `--decay-epochs`: Epoch interval for learning rate decay.
- `--warmup-epochs`: Warmup epochs.
- `--warmup-prefix`: Exclude warmup from decay.
- `--cooldown-epochs`: Cooldown epochs.
- `--patience-epochs`: Patience for Plateau scheduler.
- `--decay-rate`: Decay rate for learning rate (default: `0.1`).

#### Model Exponential Moving Average Parameters:
- `--model-ema`: Track model weight EMA.
- `--model-ema-force-cpu`: Track EMA on CPU.
- `--model-ema-decay`: EMA decay factor (default: `0.9998`).

#### Miscellaneous Parameters:
- `--seed`: Random seed (default: `42`).
- `--resume`: Resume training from checkpoint.
- `--accelerator`: Accelerator device (`cpu`, `cuda`, `tpu`, `mps`).
- `--num_workers`: Number of data loader workers (default: `4`).
- `--distributed-backend`: Distributed backend (`ddp`, `ddp2`, `horovod`, etc.).
- `--sync-bn`: Use synchronized BatchNorm.
- `--num_nodes`: Number of nodes.
- `--num_devices`: Number of devices per node.
- `--worker-seeding`: Worker seed mode (`all`, default).
- `--checkpoint-hist`: Number of checkpoints to keep.
- `--val-metric-mode`: Validation metric mode (`min`, `max`).
- `--log-interval`: Logging interval (default: `1`).
- `--amp`: Mixed precision training (default: `False`).
- `--pin-mem`: Pin memory for DataLoader.
- `--no-prefetcher`: Disable fast prefetcher.
- `--output`: Output directory for checkpoints and

 logs.
- `--eval-metric`: Evaluation metric (default: `val_loss`).

### YAML Configuration

To specify arguments via a YAML file, use the `--config` flag:

```bash
python train.py --config config.yaml
```

## Hierarchical Loss Functions

The model can utilize Hierarchical Cross Entropy (HXE) and Soft Labels to leverage hierarchical relationships between codons, as follows:

### Hierarchical Cross Entropy (HXE) with Alpha
HXE leverages hierarchical relationships by assigning weights to classification errors based on their position within a hierarchy. The parameter **alpha** controls how quickly the penalties decrease as you move down the hierarchy levels. A higher alpha value causes errors at higher hierarchy levels to be penalized more heavily, encouraging the model to avoid major mistakes that involve misclassifying entire functional groups of codons.

In a random hierarchy (selected via the `--hierarchy` parameter), the biological significance is not considered, but this can aid in exploring how the model generalizes without relying on predefined biological structures.

### Soft Labels with Beta
Soft Labels smooth out the target distribution by distributing some probability mass to classes similar to the target class. The **beta** parameter controls the degree of smoothing, with higher beta values creating softer distributions. This technique allows the model to learn similarities within hierarchical classes, enabling it to make "better" mistakes by assigning higher probabilities to codons within the same or related functional groups.

These techniques are based on the principles discussed in [“Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks”](https://arxiv.org/abs/2012.00652), where the authors show how leveraging hierarchical class structures can improve model robustness.

### Hierarchy Creation
The code can create either a true biological hierarchy or a random hierarchy. In the true hierarchy, codons are grouped based on their corresponding amino acids and biological roles (start, stop, or amino acid codons). With the `random` setting in the `--hierarchy` parameter, a random structure is generated without regard to biological significance. This feature is useful for understanding how the model behaves when codons are grouped in non-biological ways.

## Tokenization Strategies

The code supports several tokenization strategies tailored for mRNA sequence modeling:

- **3mer**: Tokenizes sequences into overlapping triplets (codons), often used for mRNA due to the biological significance of triplet codons.
- **Nucleotide (nuc)**: Treats each nucleotide (A, C, G, T) as a separate token.
- **6mer**: Uses overlapping sets of six nucleotides, offering a higher-resolution tokenization that captures longer patterns within sequences.
- **Byte Pair Encoding (BPE)**: Common in NLP, BPE merges frequent character sequences into tokens, which can be useful for mRNA sequences by learning subunit patterns.
- **WordPiece (wp)**: Splits sequences into subunits, which can capture frequent biological patterns with hierarchical encoding similar to BPE.
- **Unigram (ug)**: Generates tokens from a pre-set vocabulary, which is effective for modeling rare codon patterns by learning their likelihoods directly from the data.

### Choosing a Tokenizer
Different tokenizers may be better suited for different tasks:
- **3mer** and **nuc**: Most biologically relevant for standard mRNA modeling.
- **6mer**: Useful for capturing extended biological patterns and structures.
- **BPE** and **wp**: General-purpose, can adapt to varied sequence complexities.
- **ug**: Best for capturing patterns that have variable frequencies within the data.

## Logging and Checkpoints

Training metrics are logged using TensorBoard, with checkpoints saved based on validation performance. You can resume training from the latest checkpoint as needed.

Here’s the "Reproducing Experiments" section with environment variables for directories and explicit configurations for each experiment:

## Reproducing Experiments

To reproduce the experiments, use the following configurations for pre-training 50M parameter models with different architectures and settings. These experiments also include variations of the Hierarchical Cross Entropy (HXE) with alpha values of 0.2, 0.4, and 0.6.

### 50M Param Models Pre-training

#### GPT-2 MLM
```bash
python train.py --data_dir $DATA_DIR --train_file sequences_codonized.txt --model_name gpt2 --num_layers 10 --hidden_size 640 --batch_size 128 --intermediate_size 2560 --epochs 30 --lr 1e-3 --weight-decay 0.1 --min-lr 1e-5 --data-type codon --mode mlm --vocab_size 70 --max_seq_len 444 --output $OUTPUT_DIR/gpt2/mlm/ --num_devices $NUM_DEVICES --num_workers 16 --num_samples -1 --bidirectional --amp
```

#### GPT-2 CLM
```bash
python train.py --data_dir $DATA_DIR --train_file sequences_codonized.txt --model_name gpt2 --num_layers 10 --hidden_size 640 --batch_size 128 --intermediate_size 2560 --epochs 30 --lr 1e-3 --weight-decay 0.1 --min-lr 1e-5 --data-type codon --mode clm --vocab_size 70 --max_seq_len 444 --output $OUTPUT_DIR/gpt2/clm/ --num_devices $NUM_DEVICES --num_workers 16 --num_samples -1 --amp
```

#### Bi-Mamba MLM
```bash
python train.py --data_dir $DATA_DIR --train_file sequences_codonized.txt --model_name mamba --num_layers 20 --hidden_size 256 --batch_size 128 --intermediate_size 1024 --epochs 30 --lr 1e-3 --weight-decay 0.1 --min-lr 1e-5 --data-type codon --mode mlm --vocab_size 70 --max_seq_len 444 --output $OUTPUT_DIR/mamba/mlm/ --num_devices $NUM_DEVICES  --num_workers 16 --num_samples -1 --pos_embedding absolute --amp --bidirectional
```

#### Mamba CLM
```bash
python train.py --data_dir $DATA_DIR --train_file sequences_codonized.txt --model_name mamba --num_layers 40 --hidden_size 256 --batch_size 128 --intermediate_size 1024 --epochs 30 --lr 1e-3 --weight-decay 0.1 --min-lr 1e-5 --data-type codon --mode clm --vocab_size 70 --max_seq_len 444 --output $OUTPUT_DIR/mamba/clm/ --num_devices $NUM_DEVICES --num_workers 16 --num_samples -1 --pos_embedding absolute --amp
```

#### Hyena MLM
```bash
python train.py --data_dir $DATA_DIR --train_file sequences_codonized.txt --model_name hyena --num_layers 7 --hidden_size 768 --batch_size 128 --intermediate_size 3072 --epochs 30 --lr 1e-4 --weight-decay 0.1 --min-lr 1e-6 --data-type codon --mode mlm --vocab_size 70 --max_seq_len 444 --output $OUTPUT_DIR/hyena/mlm/ --num_devices $NUM_DEVICES --num_workers 16 --num_samples -1 --pos_embedding absolute --amp --bidirectional
```

#### Hyena CLM
```bash
python train.py --data_dir $DATA_DIR --train_file sequences_codonized.txt --model_name hyena --num_layers 7 --hidden_size 768 --batch_size 128 --intermediate_size 3072 --epochs 30 --lr 1e-4 --weight-decay 0.1 --min-lr 1e-6 --data-type codon --mode clm --vocab_size 70 --max_seq_len 444 --output $OUTPUT_DIR/hyena/clm/ --num_devices $NUM_DEVICES --num_workers 16 --num_samples -1 --pos_embedding absolute --amp
```

### Alternative Model Configurations

To adjust for different model sizes, set the following configurations:

#### 20M Parameters
```bash
--num_layers 12 --hidden_size 384 --intermediate_size 1536
```

#### 100M Parameters
```bash
--num_layers 14 --hidden_size 768 --intermediate_size 3072
```

#### 150M Parameters
```bash
--num_layers 12 --hidden_size 1024 --intermediate_size 4096
```

All experiments can be run with `--loss` of HXE and `--hxe-alpha` values of 0.2, 0.4, and 0.6 to explore the impact of hierarchical loss settings on model performance.


This section allows users to reproduce the experiments using environment variables for directories and includes configurations for different model parameter sizes without having to reference separate configuration details.

## After Training and Evaluation on Downstream Tasks

Once the model training is complete, you can proceed to evaluate its performance on downstream tasks. Detailed instructions and scripts for these tasks are provided in the `downstream` folder, which includes a separate README with guidance on running evaluations, interpreting results, and adjusting configurations for specific tasks.

For more information, refer to the [Downstream README](./downstream/README.md).

## Acknowledgements

Thanks and shoutout to the `timm` deep learning library created by Ross Wightman for providing a wide range of custom optimizers and training scripts that have greatly facilitated the development and fine-tuning of models in this project. The versatility and robustness of `timm` have been invaluable for efficient model experimentation and optimization.

## Citation
If you found this work useful, please consider citing

```bibtex
@inproceedings{yazdani2025helm,
    title={{HELM}: Hierarchical Encoding for m{RNA} Language Modeling},
    author={Mehdi Yazdani-Jahromi and Mangal Prakash and Tommaso Mansi and Artem Moskalev and Rui Liao},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=MMHqnUOnl0}
}
```