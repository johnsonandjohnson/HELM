import lightning as L
import torch
from utils import AA_TO_ID
from losses import DistillationLoss
import numpy as np
from soft_label_utils import soft_labels, classes, make_batch_soft_labels

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

CODON_TO_AA = {codon: aa if aa in AA_TO_ID else AA_TO_ID["<unk>"] for codon, aa in zip(codon_list, aa_list)}
CODON_TO_AA["<pad>"] = "<pad>"
CODON_TO_AA["<cls>"] = "<cls>"
CODON_TO_AA["<eos>"] = "<eos>"
CODON_TO_AA["<unk>"] = "<unk>"
CODON_TO_AA["<distill_token>"] = "<distill_token>"
CODON_TO_AA["<mask>"] = "<mask>"


class AntibodyLLMLightening(L.LightningModule):
    def __init__(self, model, train_mode, distill_mode, distill_type, distill_alpha, distill_temp, teacher_model, criterion, optimizer, scheduler, label_based_metrics, logits_based_metrics, token_to_id, id_to_token, aa_loss=None, tree=None, soft_labels=None):
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
        self.id_to_token = id_to_token
        self.aa_loss = aa_loss
        self.soft_labels = soft_labels
        if distill_mode:
            self.criterion_distill = DistillationLoss(self.criterion, self.teacher_model, self.distill_type, self.distill_temp)

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
        
        res = self.model(input_ids, pos_ids, masked_tokens=masked_tokens)

        if self.aa_loss:
            aa_res = torch.zeros(input_ids.shape[0], input_ids.shape[1], len(AA_TO_ID)).to(self.device)
            res_prob = torch.softmax(res.logits, dim=-1)
            for i in range(70):
                    aa_res[:, :, AA_TO_ID[CODON_TO_AA[self.id_to_token[i]]]] += res_prob[:, :, i]
            return res, aa_res
        
        return res

    def calculate_loss(self, input_ids, outputs, targets, outputs_aa=None, targets_aa=None):

        if self.train_mode == "mlm":
            mask_idx = input_ids == self.mask_token
            if self.soft_labels is not None:
                outputs_logits = torch.nn.functional.log_softmax(outputs.logits, dim=1)
                target_distribution = make_batch_soft_labels(self.soft_labels.to(self.device), targets[mask_idx].view(-1), len(classes), outputs_logits.size(0), self.device)
                loss = self.criterion(outputs_logits, target_distribution)
            else:
                loss = self.criterion(outputs.logits, targets[mask_idx].view(-1))
                if self.aa_loss:
                    aa_loss = self.criterion(outputs_aa, targets_aa.view(-1))
                    if self.aa_loss == "add":
                        loss = loss + aa_loss
                    elif self.aa_loss == "only":
                        loss = aa_loss            

        elif self.train_mode == "clm":
            if self.soft_labels is not None:
                outputs_logits = torch.nn.functional.log_softmax(outputs.logits, dim=2)
                target_distribution = make_batch_soft_labels(self.soft_labels.to(self.device), targets.view(-1), len(classes), outputs_logits.size(0) * outputs_logits.size(1), self.device)
                loss = self.criterion(outputs_logits.view(-1, self.model.vocab_size), target_distribution)
            else:
                loss = self.criterion(outputs.logits.view(-1, self.model.vocab_size), targets.view(-1))
                if self.aa_loss:
                    aa_loss = self.criterion(outputs_aa.view(-1, len(AA_TO_ID)), targets_aa.view(-1))
                    if self.aa_loss == "add":
                        loss = loss + aa_loss
                    elif self.aa_loss == "only":
                        loss = aa_loss

        if self.distill_mode:
            distillation_loss = self.criterion_distill(input_ids, outputs.distill_logits)
            loss = loss * (1 - self.distill_alpha) + distillation_loss * self.distill_alpha

        return loss
    
    def training_step(self, batch, batch_idx):
        input_ids, pos_ids, targets = batch
        if self.aa_loss:
            outputs, outputs_aa = self(input_ids, pos_ids)
            target_aa = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.long).to(self.device)
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    target_aa[i, j] = AA_TO_ID[CODON_TO_AA[self.id_to_token[targets[i, j].item()]]]
            loss = self.calculate_loss(input_ids, outputs, targets, outputs_aa, target_aa)

        else:
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
        out_labels = torch.argmax(outputs.logits, dim=-1)
        if self.train_mode == "mlm":
            mask_idx = input_ids == self.mask_token
            targets = targets[mask_idx].view(-1)
        metrics1 = self.label_based_metrics(out_labels, targets)
        if self.train_mode == "clm":
            metrics2 = self.logits_based_metrics(outputs.logits, targets)
        else:
            metrics2 = {}
        return metrics1, metrics2

    def validation_step(self, batch, batch_idx):
        input_ids, pos_ids, targets = batch
        if self.aa_loss:
            outputs, outputs_aa = self(input_ids, pos_ids)
            target_aa = torch.zeros(input_ids.shape[0], input_ids.shape[1], dtype=torch.long).to(self.device)
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    target_aa[i, j] = AA_TO_ID[CODON_TO_AA[self.id_to_token[targets[i, j].item()]]]
            loss = self.calculate_loss(input_ids, outputs, targets, outputs_aa, target_aa)
        else:
            outputs = self(input_ids, pos_ids)
            loss = self.calculate_loss(input_ids, outputs, targets)
        metrics1, metrics2 = self.get_metrics(input_ids, outputs, targets)
        val_loss = {"val_loss": loss}
        log = {**val_loss, **metrics1, **metrics2}
        self.log_dict(log, prog_bar=True, sync_dist=True, on_epoch=True)
        return loss, metrics1, metrics2
    

    