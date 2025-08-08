## Model Configurations

The `model_configs` directory contains JSON files used for setting up various models, specifying their architecture, tokenization, and training modes. These configurations define the parameters needed for initializing models in downstream tasks, making it easier to experiment with different setups.

- **HELM**: Contains model configurations designed for the HELM pretraining approach, focusing on different tokenization methods and model architectures.
- **XE**: Consists of configurations for cross-entropy-based training with various models and tokenization strategies, including models like Hyena, Mamba, and transformers.

#### JSON File Structure

Each JSON configuration file outlines the model's architecture, mode, and training parameters. Here’s a breakdown of a typical JSON file structure:

```json
{
    "model_type": "transformer",
    "mode": "mlm",
    "tokenizer": "3mer",
    "helm": true,
    "model_config": {
        "model": "gpt2",
        "d_model": 640,
        "n_layer": 10,
        "d_intermediate": 2560,
        "pos_embedding": "absolute",
        "max_position_embeddings": 2048,
        "bidirectional": true
    }
}
```

- **model_type**: Specifies the type of model, such as "transformer," "hyena," or "mamba."
- **mode**: Defines the training mode, e.g., "mlm" (masked language model) or "clm" (causal language model).
- **tokenizer**: Indicates the tokenization strategy, like "3mer" or "nuc" for nucleotide-level tokenization.
- **helm**: A boolean indicating whether the HELM pretraining approach is used.
- **model_config**: A nested dictionary specifying model architecture parameters:
  - **model**: Base model architecture (e.g., "gpt2").
  - **d_model**: Dimensionality of model embeddings.
  - **n_layer**: Number of layers in the model.
  - **d_intermediate**: Size of the intermediate layer.
  - **pos_embedding**: Type of positional embedding used (e.g., "absolute").
  - **max_position_embeddings**: Maximum number of positions supported by the model.
  - **bidirectional**: Boolean indicating whether the model is bidirectional (used in masked language modeling).

These configuration files enable flexible experimentation with different model settings, simplifying the setup process for downstream tasks. Adjustments can be made directly in the JSON files to modify model behavior, architecture, and training objectives.
