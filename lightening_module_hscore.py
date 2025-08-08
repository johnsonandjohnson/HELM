import lightning as L
import torch
from utils import AA_TO_ID
from losses import DistillationLoss
import numpy as np

def build_parent_dict(hierarchy):
    parent_dict = {}
    for parent, children in hierarchy.items():
        for child in children:
            parent_dict[child] = parent
    return parent_dict

def get_path_to_root(node, parent_dict):
    path = []
    while node in parent_dict:
        path.append(node)
        node = parent_dict[node]
    path.append(node)  # Append the root
    return path[::-1]

def find_lca(node1, node2, parent_dict):
    path1 = get_path_to_root(node1, parent_dict)
    path2 = get_path_to_root(node2, parent_dict)
    lca = None
    for ancestor1, ancestor2 in zip(path1, path2):
        if ancestor1 == ancestor2:
            lca = ancestor1
        else:
            break
    return lca

def height_of_node(node, parent_dict):
    height = 0
    while node in parent_dict:
        height += 1
        node = parent_dict[node]
    return height

def hierarchical_distance_metric(true_class, predicted_class, hierarchy):
    parent_dict = build_parent_dict(hierarchy)
    if true_class == predicted_class:
        return 0  # No misclassification
    lca = find_lca(true_class, predicted_class, parent_dict)
    if lca is None:
        return -1  # No common ancestor found (shouldn't happen in a valid hierarchy)
    height_lca = height_of_node(lca, parent_dict)
    
    if height_lca == 1:
        return 3
    elif height_lca == 2:
        return 2
    elif height_lca == 3:
        return 1
    return height_lca

def get_avg_h_score(x, y, hierarchy):
    h_scores = []
    for true_class, predicted_class in zip(x, y):
        if true_class == predicted_class:
            continue
        h_score = hierarchical_distance_metric(true_class, predicted_class, hierarchy)
        h_scores.append(h_score)
    avg_score = np.mean(h_scores)
    dict_count = {}
    for item in h_scores:
        if item in dict_count:
            dict_count[item] += 1
        else:
            dict_count[item] = 1

    return avg_score, dict_count


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def build_parent_dict(hierarchy):
    parent_dict = {}
    for parent, children in hierarchy.items():
        for child in children:
            parent_dict[child] = parent
    return parent_dict

def get_path_to_root(node, parent_dict):
    path = []
    while node in parent_dict:
        path.append(node)
        node = parent_dict[node]
    path.append(node)  # Append the root
    return path[::-1]

def tree_edit_distance(true_tokens, predicted_tokens, hierarchy):
    parent_dict = build_parent_dict(hierarchy)
    total_distance = 0
    num = 0
    for true, pred in zip(true_tokens, predicted_tokens):
        true_path = get_path_to_root(true, parent_dict)
        pred_path = get_path_to_root(pred, parent_dict)
        total_distance += levenshtein_distance(true_path, pred_path)

    ted = total_distance / len(true_tokens)
    return ted

hierarchy = {
    'V': ["GTT", "GTC", "GTA", "GTG"],
    'A': ["GCT", "GCC", "GCA", "GCG"],
    'D': ["GAT", "GAC"],
    'E': ["GAA", "GAG"],
    'G': ["GGT", "GGC", "GGA", "GGG"],
    'F': ["TTT", "TTC"],
    'L': ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    'S': ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    'Y': ["TAT", "TAC"],
    'C': ["TGT", "TGC"],
    'W': ["TGG"],
    'P': ["CCT", "CCC", "CCA", "CCG"],
    'H': ["CAT", "CAC"],
    'Q': ["CAA", "CAG"],
    'R': ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    'I': ["ATT", "ATC", "ATA"],
    'T': ["ACT", "ACC", "ACA", "ACG"],
    'N': ["AAT", "AAC"],
    'K': ["AAA", "AAG"],
    'start_codon': ["ATG"],
    'end_codon': ["TAA", "TAG", "TGA"],
    'aa_codons': ['V', 'A', 'D', 'E', 'G', 'F', 'L', 'S', 'Y', 'C', 'W', 'P', 'H', 'Q', 'R', 'I', 'T', 'N', 'K'],
    'codons': ['start_codon', 'end_codon', 'aa_codons'],
    'pad': ["<pad>"],
    'cls': ["<cls>"],
    'eos': ["<eos>"],
    'unk': ["<unk>"],
    'mask': ["<mask>"],
    'distill_token': ["<distill_token>"],
    'Root': ['codons', 'pad', 'cls', 'eos', 'unk', 'mask', 'distill_token']
}


class AntibodyLLMLightening(L.LightningModule):
    def __init__(self, model, train_mode, distill_mode, distill_type, distill_alpha, distill_temp, teacher_model, criterion, optimizer, scheduler, label_based_metrics, logits_based_metrics, token_to_id, tree=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label_based_metrics = label_based_metrics
        self.logits_based_metrics = logits_based_metrics
        self.train_mode = train_mode
        self.distill_mode = distill_mode
        self.distill_type = distill_type
        self.distill_alpha = distill_alpha
        self.distill_temp = distill_temp
        self.teacher_model = teacher_model
        self.token_to_id = token_to_id
        self.mask_token = self.token_to_id["<mask>"] if tree is None else tree.index("<mask>")
        self.pad_token = self.token_to_id["<pad>"] if tree is None else tree.index("<pad>")
        if distill_mode:
            self.criterion_distill = DistillationLoss(self.criterion, self.teacher_model, self.distill_type, self.distill_temp)
        self.tree = tree
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.true_counter = {}
        

    def get_distillation_sequence(self, input_ids, pos_ids):
        if self.distillation_mode == "all_tokens":
            input_ids_new = torch.empty((input_ids.shape[0], input_ids.shape[1] * 2), dtype=torch.long)
            pos_ids_new = torch.empty((pos_ids.shape[0], pos_ids.shape[1] * 2), dtype=torch.long)
            input_ids_new[:, 1::2] = self.token_to_id["<distill_token>"]
            pos_ids_new[:, 1::2] = pos_ids
            input_ids_new[:, ::2] = input_ids
            pos_ids_new[:, ::2] = pos_ids
        elif self.distill_mode == "last_token":
            input_ids_new = torch.empty((input_ids.shape[0], input_ids.shape[1] + 1), dtype=torch.long)
            pos_ids_new = torch.empty((pos_ids.shape[0], pos_ids.shape[1] + 1), dtype=torch.long)
            input_ids_new[:, :-1] = input_ids
            pos_ids_new[:, :-1] = pos_ids
            input_ids_new[:, -1] = self.token_to_id["<distill_token>"]
            pos_ids_new[:, -1] = pos_ids[:, -1] + 1
        else:
            return input_ids, pos_ids
        
        return input_ids_new, pos_ids_new
    
    def forward(self, input_ids, pos_ids):

        if self.distill_mode:
            input_ids, pos_ids = self.get_distillation_sequence(input_ids, pos_ids)
        
        if self.train_mode == "mlm":
            masked_tokens = input_ids == self.mask_token
            masked_tokens = masked_tokens & (input_ids != self.pad_token)

        else:
            masked_tokens = None        

        return self.model(input_ids, pos_ids, masked_tokens=masked_tokens)

    def calculate_loss(self, input_ids, outputs, targets):

        if self.train_mode == "mlm":
            mask_idx = input_ids == self.mask_token
            loss = self.criterion(outputs.logits, targets[mask_idx].view(-1))

        elif self.train_mode == "next_token_pred":
            loss = self.criterion(outputs.logits.view(-1, self.model.vocab_size), targets.view(-1))

        if self.distill_mode:
            distillation_loss = self.criterion_distill(input_ids, outputs.distill_logits)
            loss = loss * (1 - self.distill_alpha) + distillation_loss * self.distill_alpha

        return loss
    
    def training_step(self, batch, batch_idx):
        input_ids, pos_ids, targets = batch
        outputs = self(input_ids, pos_ids)
        loss = self.calculate_loss(input_ids, outputs, targets)
        metrics1, metrics2 = self.get_metrics(input_ids, outputs, targets)
        log = {'train_loss': loss, 'lr': self.optimizer.param_groups[-1]['lr'], **metrics1, **metrics2}
        self.log_dict(log, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        else:
            return self.optimizer
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)

    def get_metrics(self, input_ids, outputs, targets):
        
        if self.train_mode == "mlm":
            mask_idx = input_ids == self.mask_token
            targets = targets[mask_idx].view(-1)
        
        out_labels = torch.argmax(outputs.logits, dim=-1)
        out_labels_all = out_labels.flatten()
        targets_all = targets.flatten()
        out_labels_all_new = []
        targets_all_new = []
        for i in range(out_labels_all.shape[0]):
            if self.tree:
                out_labels_all_new.append(self.tree[out_labels_all[i].item()])
                targets_all_new.append(self.tree[targets_all[i].item()])
            else:
                out_labels_all_new.append(self.id_to_token[out_labels_all[i].item()])
                targets_all_new.append(self.id_to_token[targets_all[i].item()])

        metrics1 = self.label_based_metrics(out_labels, targets)
        if self.train_mode == "next_token_pred":
            metrics2 = self.logits_based_metrics(outputs.logits, targets)
        else:
            metrics2 = {}
        h_score, dict_count = get_avg_h_score(out_labels_all_new, targets_all_new, hierarchy)
        for key in dict_count.keys():
            if key in self.true_counter:
                self.true_counter[key] += dict_count[key]
            else:
                self.true_counter[key] = dict_count[key]

        return metrics1, metrics2, h_score , tree_edit_distance(out_labels_all_new, targets_all_new, hierarchy)

    def validation_step(self, batch, batch_idx):
        input_ids, pos_ids, targets = batch
        outputs = self(input_ids, pos_ids)
        loss = self.calculate_loss(input_ids, outputs, targets)
        metrics1, metrics2, metrics3, metrics4 = self.get_metrics(input_ids, outputs, targets)
        val_loss = {"val_loss": loss}
        h_score = {"h_score": metrics3}
        t_edit_score = {"t_edit_score": metrics4}
        log = {**val_loss, **metrics1, **metrics2, **h_score, **t_edit_score}
        self.log_dict(log, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss, outputs, input_ids, metrics1, metrics2
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, pos_ids, targets = batch
        outputs = self(input_ids, pos_ids)
        metrics1, metrics2, metrics3, metrics4 = self.get_metrics(input_ids, outputs, targets)
        return outputs.logits.float().cpu().numpy(), targets.cpu().numpy()
    

    