# Build model to work with pretrained embeddings for downstream tasks

import torch
import os
import torch.nn as nn
import numpy as np
from evaluation_metrics import calculate_pearson_r, calculate_spearman_rank_corr, calculate_r_squared, calculate_accuracy, calculate_accuracy_multi, calculate_f1_score_multi
import torch.nn.functional as F
from tqdm import tqdm

class OneHotBaseline(nn.Module):
    def __init__(self):
        super(OneHotBaseline, self).__init__()
        
        # needed to keep track of the device of a model
        self.dummy_parameter = torch.nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        oh_x = nn.functional.one_hot(x).float()
        return oh_x


class BaseNNModel(nn.Module):
    def __init__(self):
        super(BaseNNModel, self).__init__()

    def train_model(self, train_loader, val_loader, test_loader, criterion, optimizer, device, task_type, num_epochs,
                    early_stopping_patience, scheduler, grad_accumulation, output_size, metric, save_dir):
        best_val_loss = float("inf")
        best_val_metric = float("-inf")
        best_f1 = float("-inf")
        best_f1_test = float("-inf")
        best_f1_test_test = float("-inf")
        best_f1_test_val = float("-inf")
        best_val_test_metric = float("-inf")
        best_test_metric = float("-inf")
        best_test_val_metric = float("-inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            i = 0
            for inputs, pos_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training"):
                inputs, pos_ids, labels = inputs.to(device), pos_ids.to(device), labels.to(device)
                outputs = self(inputs, pos_ids).to(torch.float32)
                labels = labels.to(torch.float32)
                if task_type == "segmentation":
                    outputs = outputs.view(labels.shape[0], labels.shape[1], -1)
                    labels = labels.view(-1, labels.shape[1]).type(torch.LongTensor)
                    loss = 0
                    for i in range(outputs.shape[1]):
                        loss += criterion(outputs[:, i, :].to(device), labels[:, i].to(device))
                elif task_type == "classification":
                    labels = labels.view(-1).type(torch.LongTensor)
                    loss = criterion(outputs.to(device), labels.to(device))
                else:
                    loss = criterion(outputs.to(device), labels.view(-1, output_size).to(device))

                loss.backward()
                if (i+1) % grad_accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
                i += 1

            self.eval()
            val_loss = 0.0

            with torch.no_grad():
                if task_type == "segmentation":
                    result, f1, _, _, val_loss = self.evaluate_model(val_loader, task_type, device, output_size, metric, criterion)
                    result_test,f1_test, _, _ , val_loss= self.evaluate_model(test_loader, task_type, device, output_size, metric, criterion)
                    print(f"Test {metric}: {result_test}, Test F1: {f1_test}")
                else:
                    result, _, _ , val_loss= self.evaluate_model(val_loader, task_type, device, output_size, metric, criterion)
                    result_test, _, _, val_loss = self.evaluate_model(test_loader, task_type, device, output_size, metric, criterion)
                    print(f"Test {metric}: {result_test}")

            if scheduler is not None:
                if task_type == "segmentation":
                    scheduler.step(val_loss)
                else:
                    scheduler.step(result)

            # Check for early stopping
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)

#             if val_loss < best_val_loss:
            if np.mean(result) > np.mean(best_val_metric):
#                 best_val_loss = val_loss
                best_val_metric = result
                if task_type == "segmentation":
                    best_f1 = f1
                    best_f1_test = f1_test

                best_val_test_metric = result_test
                patience_counter = 0
                best_model_weights = self.state_dict()
                print(f"Best model weights found!")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
            
            if np.mean(result_test) > np.mean(best_test_metric):
                best_test_metric = result_test
                best_test_val_metric = result
                if task_type == "segmentation":
                    best_f1_test_test = f1_test
                    best_f1_test_val = f1
                

            if task_type == "segmentation":
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val {metric}: {result}, Val F1: {f1}")
                
            elif task_type == "classification":
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val {metric}: {result}%")
            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val {metric}: {result}")
                
        print("Saving best model weights at:", os.path.join(save_dir, "best_model_weights.pth"))
        torch.save(best_model_weights, os.path.join(save_dir, "best_model_weights.pth"))
        if task_type == "segmentation":
            return best_val_metric, best_val_test_metric, best_f1, best_f1_test, best_test_metric, best_test_val_metric,  best_f1_test_test, best_f1_test_val
        else:
            return best_val_metric, best_val_test_metric,  best_test_metric, best_test_val_metric


    def evaluate_model(self, dataloader, task_type, device, output_size, metric, criterion):
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            self.eval()
            val_loss = 0.0
            for inputs, pos_ids, labels in tqdm(dataloader, desc=f"Evaluation"):
                inputs, pos_ids, labels = inputs.to(device), pos_ids.to(device), labels.to(device)
                outputs = self(inputs, pos_ids).to(torch.float32)
                labels = labels.to(torch.float32)
                if task_type == "segmentation":
                    outputs = outputs.view(labels.shape[0], labels.shape[1], -1)
                    labels = labels.view(-1, labels.shape[1]).type(torch.LongTensor)
                    loss = 0
                    for i in range(outputs.shape[1]):
                        loss += criterion(outputs[:, i, :].to(device), labels[:, i].to(device))
                elif task_type == "classification":
                    labels = labels.view(-1).type(torch.LongTensor)
                    loss = criterion(outputs.to(device), labels.to(device))
                else:
                    loss = criterion(outputs.to(device), labels.view(-1, output_size).to(device))
                val_loss += loss.item()
                if task_type == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.append(predicted.cpu().numpy())
                elif task_type == 'segmentation':
                    outputs = outputs.view(labels.shape[0], labels.shape[1], -1)
                    _, predicted = torch.max(outputs.data, 2)
                    all_predictions.append(predicted.cpu().numpy())
                    
                else:
                    all_predictions.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        predictions_flat = np.concatenate(all_predictions, axis=0)
        targets_flat = np.concatenate(all_targets, axis=0)

        if task_type == 'classification':
            return calculate_accuracy(targets_flat, predictions_flat),  predictions_flat, targets_flat, loss
        elif task_type == 'segmentation':
            return calculate_accuracy_multi(targets_flat, predictions_flat), calculate_f1_score_multi(targets_flat, predictions_flat), predictions_flat, targets_flat, loss
        else:
            if metric == "pearson":
                return calculate_pearson_r(targets_flat, predictions_flat), predictions_flat, targets_flat, loss
            if metric == "spearman":
                return calculate_spearman_rank_corr(targets_flat, predictions_flat), predictions_flat, targets_flat, loss
            if metric == "r_squared":
                return calculate_r_squared(targets_flat, predictions_flat), predictions_flat, targets_flat, loss

class TextCNN(BaseNNModel):
    
    def __init__(self, input_size, embed_size, num_classes, embed_input, conv_size=100, kernel_sizes=[3,4,5], dropout_rate=0.2):
        super(TextCNN, self).__init__()
        """
        
        input_size (int): dimension of input embedding
        num_classes (int): dimension of output embedding
        embed_size (int): embed input features into input_size -> embed_size via Linear layer
        embed_input (bool): if True then input embedding is computed via Linear layer; if False then uses input features as is
        conv_size (int): dim of conv features
        kernel_sizes (list): spatial extent along n_nucleotides dimension
        """
        
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.embed_input = embed_input # if True then assumes already embeded inputs
                
        # text-cnn params
        self.conv_size = conv_size
        self.kernel_sizes = kernel_sizes
        
        # embed layer
        if embed_input:
            self.embed_size = embed_size
            self.embed = nn.Linear(input_size, embed_size)

        else:
            self.embed_size = input_size
        
        # text-cnn layers
        self.convs = nn.ModuleList([nn.Conv2d(1, conv_size, [ks, embed_size]) for ks in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes)*conv_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        # x: (batch, n_nucleotides, input_dim)
        # embed
        if self.embed_input:
            x = self.embed(x) # input_dim -> embed_size
            
        # text-cnn
        x = x.unsqueeze(1)
        x = [nn.functional.relu(conv(x)).squeeze(-1) for conv in self.convs]
        x = [nn.functional.max_pool1d(xc, xc.size(-1)).squeeze(-1) for xc in x]
        x = torch.cat(x, -1)
        x = self.dropout(x)
        
        x = self.fc(x)
        
        return x
    

class SupervisedFinetuneHFModel(BaseNNModel):
    def __init__(self, backbone, head, backbone_type, pad_id, fine_tune=False, return_masked_logits=False):
        super(SupervisedFinetuneHFModel, self).__init__()
        
        self.backbone = backbone
        self.head = head
        self.backbone_type = backbone_type
        self.return_masked_logits = return_masked_logits
        self.fine_tune = fine_tune
        self.pad_id = pad_id
        if fine_tune == False:
            self.backbone.eval()
                
    def forward(self, x, pos):
        tokens_ids = x
        attention_mask_pad = tokens_ids != self.pad_id 
        attention_mask = attention_mask_pad
        if self.fine_tune:
            self.backbone.train()
            if self.backbone_type == "transformer":
                model_output = self.backbone(input_ids=tokens_ids,
                                        position_ids=pos, attention_mask=attention_mask)
                token_embeddings = model_output.last_hidden_state
            elif self.backbone_type == "mamba":
                model_output = self.backbone(tokens_ids,
                                        position_ids=pos)
                token_embeddings = model_output
            elif self.backbone_type == "hyena":
                model_output = self.backbone(tokens_ids)
                token_embeddings = model_output[0]
                
        else:
            with torch.no_grad():
                self.backbone.eval()
                if self.backbone_type == "transformer":
                    model_output = self.backbone(input_ids=tokens_ids,
                                            position_ids=pos, attention_mask=attention_mask)
                    token_embeddings = model_output.last_hidden_state
                elif self.backbone_type == "mamba":
                    model_output = self.backbone(tokens_ids,
                                            position_ids=pos)
                    token_embeddings = model_output
                
                elif self.backbone_type == "hyena":
                    model_output = self.backbone(tokens_ids)
                    token_embeddings = model_output[0]

        # head predict
        predictions = self.head(token_embeddings)
    
        return predictions
    