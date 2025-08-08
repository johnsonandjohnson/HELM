import argparse
import os
from downstream_model import SupervisedFinetuneHFModel, TextCNN
from model_factory import create_model
import json
import torch
from peft import LoftQConfig, LoraConfig, get_peft_model
from helpers import get_numpy_value, set_seeds, prepare_data, load_config

def run_experiment(gpu_id, param_config):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    dataset_config, seed_index, model_config, model_path, finetune, batch_size, lr, lr_2 = param_config
    model_config_path = model_config
    model_config = load_config(model_config)
    dataset_config = load_config(dataset_config)
    seeds = len(dataset_config["path"].keys())
    helm = model_config["helm"]

    if seed_index >= seeds:
        seed_index = 0
    seed_key = list(dataset_config["path"].keys())[seed_index]
    data_train, data_val, data_test = prepare_data(dataset_config["path"][seed_key]["path_train"],
                                                   dataset_config["path"][seed_key]["path_val"],
                                                   dataset_config["path"][seed_key]["path_test"],
                                                   model_config["tokenizer"], dataset_config["data_column"],
                                                   dataset_config["target_column"], helm)
    mode = model_config["mode"]
    print(f"GPU {gpu_id}: Model: {model_config['model_type']}, Mode: {mode}, Finetune: {finetune} trained on {dataset_config['path'][seed_key]['path_train']}")

    print(f"GPU {gpu_id}: Batch Size: {batch_size}, LR Head: {lr}, LR Backbone: {lr_2}")
    
    set_seeds()  # Set seeds for reproducibility

    model, token_ids, lora_target_modules = create_model(model_config, model_path)
    pad_id = token_ids[1]
    model.to(device)

    if dataset_config["task"] == "regression":
        head_out_features = 1
    elif "oas" in dataset_config["path"][seed_key]["path_train"]:
        head_out_features = 6314
    
    d_model = model_config["model_config"]["d_model"]
    model_type = model_config["model_type"]
    head = TextCNN(d_model, d_model * 2, head_out_features, True)
    if finetune == "lora":
        loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=1)           # set 4bit quantization
        lora_config = LoraConfig(peft_type="LORA", target_modules=lora_target_modules, use_rslora=True, loftq_config=loftq_config)
        peft_model = get_peft_model(model, lora_config)

    model_downstream = SupervisedFinetuneHFModel(
        backbone=model if finetune != "lora" else peft_model,
        head=head,
        backbone_type=model_type,
        pad_id=None if model_type!="transformer" else pad_id,
        return_masked_logits=False,
        fine_tune=False if finetune == "none" else True
    )

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
    test_data_loader = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
    
    # Initialize optimizer and scheduler
    base_params = [p for n, p in model_downstream.named_parameters() if "head" not in n]
    head_params = [p for n, p in model_downstream.named_parameters() if "head" in n]
    if finetune == "none":
        optimizer = torch.optim.Adam(head_params, lr=lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': lr_2},
            {'params': head_params, 'lr': lr}
        ])

    if "oas" in dataset_config["path"][seed_key]["path_train"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.1, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.1, verbose=True)
    
    model_downstream.to(device)
    criterion = torch.nn.CrossEntropyLoss() if dataset_config["loss"] == "xe" else torch.nn.MSELoss()
    task = dataset_config["task"] 
    metric = dataset_config["metric"]
    # Training loop
    if task  == "segmentation":
        best_val_metric, best_val_test_metric, best_f1, best_f1_test, best_test_metric, best_test_val,  best_f1_test_test, best_f1_test_val = model_downstream.train_model(
            train_dataloader, val_dataloader, test_data_loader, criterion, optimizer, device, 
            task, 300, 21, scheduler, 1, 1, metric, "./")

        return {
            "model": model_config_path,
            "mode": mode,
            "finetune": finetune,
            "val_metric": get_numpy_value(best_val_metric),
            "test_metric": get_numpy_value(best_val_test_metric),
            "val_f1": get_numpy_value(best_f1),
            "test_f1": get_numpy_value(best_f1_test),
            "test_test": get_numpy_value(best_test_metric),
            "test_val": get_numpy_value(best_test_val),
            "test_test_f1": get_numpy_value(best_f1_test_test),
            "test_val_f1": get_numpy_value(best_f1_test_val),
            "batch_size": batch_size,
            "dataset": dataset_config["path"][seed_key]["path_train"],
            "lr": lr,
            "lr_2": lr_2
        }
    else:
        best_val_metric, best_test_metric, best_test_test, best_test_val = model_downstream.train_model(
            train_dataloader, val_dataloader, test_data_loader, criterion, optimizer, device, 
            task, 300, 15, scheduler, 1, 1, metric, "./"
        )
    
        return {
            "model": model_config_path,
            "mode": mode,
            "finetune": finetune,
            "val_metric": get_numpy_value(best_val_metric),
            "test_metric": get_numpy_value(best_test_metric),
            "test_test": get_numpy_value(best_test_test),
            "test_val": get_numpy_value(best_test_val),
            "batch_size": batch_size,
            "dataset": dataset_config["path"][seed_key]["path_train"],
            "lr": lr,
            "lr_2": lr_2
        }
    

parser = argparse.ArgumentParser(description="Run experiments with various configurations")
parser.add_argument("--dataset-config", type=str, help="Path to the dataset configuration file")
parser.add_argument("--seed-index", type=int, help="Seed index to use for the dataset", default=0)
parser.add_argument("--model-config", type=str, help="Path to the model configuration file")
parser.add_argument("--model-path", type=str, help="Path to the model checkpoint")
parser.add_argument("--batch-size", type=int, choices=[8, 16, 32, 64], help="Batch size for training")
parser.add_argument("--learning-rate-head", type=float, help="Learning rate for training")
parser.add_argument("--learning-rate-backbone", type=float, help="Learning rate for training", default=0)
parser.add_argument("--finetune", type=str, choices=["none", "full", "lora"], help="Finetuning method", default="none")
parser.add_argument("--output-path", type=str, help="Path to save the results")
parser.add_argument("--gpu-id", type=int, help="GPU ID to use")

args = parser.parse_args()

if __name__ == "__main__":
    batch_size = args.batch_size
    lr = args.learning_rate_head
    lr_2 = args.learning_rate_backbone
    dataset_config = args.dataset_config
    model_path = args.model_path
    seed_index = args.seed_index
    model_config = args.model_config
    gpu_id = args.gpu_id
    file_path =  args.output_path

    dataset = json.load(open(dataset_config, 'r'))
    seeds = len(dataset["path"].keys())

    if seed_index >= seeds:
        seed_index = 0

    seed_key = list(dataset["path"].keys())[seed_index]
    dataset_name = dataset["path"][seed_key]["path_train"]

    results_all = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results_all = json.load(f)
    for item in results_all:
        if item["model"] == model_config and item["dataset"] == dataset_name and item["lr"] == lr and item["batch_size"] == batch_size:
            import sys
            sys.exit(0)
    results = []
    results.append(run_experiment(gpu_id, (dataset_config, seed_index, model_config, model_path, "none", batch_size, lr, lr_2)))
    file_path = args.output_path

    # Read existing results
    results_all = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            results_all = json.load(f)

    # Extend with new results
    results_all.extend(results)

    # Write updated results
    with open(file_path, "w") as f:
        json.dump(results_all, f)

    print(f"Completed batch size: {batch_size}, learning rate: {lr}, model: {model_config}, dataset: {dataset_name}")
    print(f"Current results: {results}")
