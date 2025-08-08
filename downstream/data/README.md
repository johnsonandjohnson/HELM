# Data
The data directory contains various datasets that can be used for downstream tasks, organized into subfolders. Each subfolder represents a distinct dataset and includes configuration files for data processing and model training. The datasets cover a wide range of topics, allowing for different types of analysis and evaluation.
## Directory Structure
```plaintext
data/
│
├── Deg/
├── Fungal/
├── iCodon/
├── MRFP/
├── OAS/
├── TcRibo/
│
└── prepare_data.py

```
## Available Datasets
Dataset configurations are organized under the `data/downstream`
- **Deg**: Contains data related to degradation or stability studies, potentially focusing on sequence modifications and their effects.
- **Fungal**: Could include fungal sequence data, useful for applications in pathogen detection or biotechnology.
- **iCodon**: (Diez et al., 2022) includes 65357 mRNA sequences with thermostability profiles from humans, mice, frogs, and fish.
- **MRFP**: (Nieuwkoop et al., 2023) has 1459 mRNA sequences with protein production levels for various gene variants in E. coli.
- **OAS**: hold-out set of 2000 antibody-encoding sequences from curated OAS data.
- **TcRibo**: (Groher et al., 2018) consists of 355 riboswitch mRNA sequences with switching factor measurements.


## Preparing New Data

The `prepare_data.py` script in this directory is responsible for processing the raw data (CSV format), splitting it into training, validation, and test sets, and generating configuration files for each dataset. It supports both embedded splits (where splits are predefined in the dataset) and custom splits based on specified ratios.
Each folder includes configuration files for different data splits and settings.


### Key Features

- Raw data preprocessing for downstream tasks.
- Supports embedded or random data splitting.
- Generates separate CSV files for train, validation, and test splits.
- Creates a JSON configuration file with dataset paths, column names, task type, and evaluation metrics.

### Usage

To run the script, use the following command:

```bash
python prepare_data.py --data-path <DATA_PATH> \
                       --output-path <OUTPUT_PATH> \
                       --embedded-split \
                       --split-column <SPLIT_COLUMN> \
                       --split-ratio <SPLIT_RATIO> \
                       --random-states <RANDOM_STATES> \
                       --data-column <DATA_COLUMN> \
                       --target-column <TARGET_COLUMN> \
                       --task <TASK> \
                       --metric <METRIC> \
                       --loss <LOSS>
```

#### Command-Line Arguments

- `--data-path`: Path to the raw CSV file.
- `--output-path`: Directory to save processed data and configuration.
- `--embedded-split`: Use if splits are already indicated in the dataset.
- `--split-column`: Column specifying the split type (e.g., 'train', 'val', 'test'). Required if `--embedded-split` is set.
- `--split-ratio`: Ratios for custom splits (e.g., `0.8 0.1 0.1`).
- `--random-states`: List of random states for reproducibility (e.g., `42 123`).
- `--data-column`: Name of the column containing sequence data.
- `--target-column`: Name of the column containing target data.
- `--task`: Type of downstream task (`regression`, `segmentation`, `classification`).
- `--metric`: Evaluation metric (`spearman`, `accuracy`).
- `--loss`: Loss function (`xe`, `mse`).

### Examples

#### Using Embedded Splits

If the dataset has a predefined split column:

```bash
python prepare_data.py --data-path /path/to/raw_data.csv \
                       --output-path /path/to/output \
                       --embedded-split \
                       --split-column split \
                       --data-column sequence \
                       --target-column label \
                       --task classification \
                       --metric accuracy \
                       --loss xe
```

#### Custom Splitting

For custom splits with specific ratios and random states:

```bash
python prepare_data.py --data-path /path/to/raw_data.csv \
                       --output-path /path/to/output \
                       --split-ratio 0.8 0.1 0.1 \
                       --random-states 42 123 \
                       --data-column sequence \
                       --target-column label \
                       --task regression \
                       --metric spearman \
                       --loss mse
```

### Output Details

The script generates:

1. **Train, Validation, and Test CSV Files**: Saved in the specified output directory.
2. **Configuration File (`config.json`)**: Includes paths to data splits, column names, task type, metric, and loss function.

#### Configuration File Example

```json
{
    "path": {
        "file_seed42": {
            "path_train": "./file_train42/0.part",
            "path_val": "./file_val42/0.part",
            "path_test": "./file_test42/0.part"
        },
        "file_seed123": {
            "path_train": "./file_train123/0.part",
            "path_val": "./file_val123/0.part",
            "path_test": "./file_test123/0.part"
        }
    },
    "data_column": "sequence",
    "target_column": "label",
    "task": "regression",
    "metric": "spearman",
    "loss": "mse"
}
```