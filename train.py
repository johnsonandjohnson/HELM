import argparse
import json
import logging
import random
import sys
import numpy as np
import torch
import yaml
from dataset import DataAntiBody
from lightening_module import AntibodyLLMLightening
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import scheduler_kwargs, create_scheduler_v2
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection
from torchmetrics.text import Perplexity
import lightning as L
from timm import utils
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, ProgressBar
from lightning.pytorch import loggers as pl_loggers
import logging
from utils import AA_TO_ID, encode_sequence_codon, encode_sequence_aa
from functools import partial


has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Antibody LLM Training')

# Data
group = parser.add_argument_group('Data')
group.add_argument('--data_dir', default='./',
                   type=str, help='Data directory')
group.add_argument('--train_file', default='SRR9179282_paired 2.csv',
                   type=str, help='Training data file')
group.add_argument('--file_extension', default='txt', type=str,
                   help='File extension')
group.add_argument('--column_name', default="sequence_heavy", type=str,
                   help='Column name for csv files')
group.add_argument('--val_file', default=None,
                   type=str, help='Validation data file')
group.add_argument('--test_file', default=None,
                   type=str, help='Test data file')
group.add_argument('--max_seq_len', default=256, type=int,
                   help='Maximum sequence length')
group.add_argument('--masking_strategy', default=['single'], type=str, nargs='+',
                   choices=['single', 'double', 'span'], help='Masking strategy')
group.add_argument('--masking_ratio', default=0.15,
                   type=float, help='Masking ratio')
group.add_argument('--data-type', default='nuc', type=str, choices=['nuc', 'aa'],
                   help='Data type')
group.add_argument('--tokenizer', default='3mer', type=str, help='Tokenizer', choices=['3mer', 'nuc', '6mer', 'bpe', 'wp', 'ug'])
group.add_argument('--overlap', default=0, type=int, help='Overlapping tokens')
group.add_argument('--position_ids', action='store_true',
                   help='Use position ids')
group.add_argument('--mode', default='clm', type=str, choices=['mlm', 'clm'],
                   help='Training mode')
group.add_argument('--batch_size', default=64, type=int,
                   help='Batch size')
group.add_argument('--num_samples', default=-1, type=int, help='Number of samples')

# Model
group = parser.add_argument_group('Model')
group.add_argument('--model_name', default='gpt2',
                   type=str, help='Model name')
group.add_argument('--bidirectional', action='store_true', default=False,
                   help='Bidirectional model')
group.add_argument('--ssm_layer', default='Mamba2', type=str, help='SSM layer')
group.add_argument('--vocab_size', default=70,
                   type=int, help='Vocabulary size')
group.add_argument('--num_layers', default=4,
                   type=int, help='Number of layers')
group.add_argument('--num_heads', default=4, type=int, help='Number of heads')
group.add_argument('--pad_vocab_size_multiple', default=1,
                   type=int, help='Pad vocab size multiple')
group.add_argument('--pos_embedding', default=None, type=str,
                   help='Position embedding type')
group.add_argument('--pos_embedding_apply', default='add', type=str,
                   help='Position embedding apply')
group.add_argument('--pos_embedding_dim', default=None, type=int,
                   help='Position embedding dimension')
group.add_argument('--tie_embeddings', action='store_true',
                   help='Tie embeddings', default=False)
group.add_argument('--hidden_size', default=256, type=int, help='Hidden size')
group.add_argument('--intermediate_size', default=1024,
                   type=int, help='Intermediate size')
group.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')
group.add_argument('--attn_dropout', default=0.0, type=float,
                   help='Attention dropout rate')
group.add_argument('--activation', default='gelu',
                   type=str, help='Activation function')

# Non-markovian relation
group.add_argument('--non_markovian_relation',
                   action='store_true', default=False, help='Non-markovian relation')
group.add_argument('--non_markovian_relation_num_heads', default=4, type=int,
                   help='Non-markovian relation number of heads')
group.add_argument('--non_markovian_relation_kernel_size',
                   default=33, type=int, help='Non-markovian relation kernel size')
group.add_argument('--non_markovian_relation_dropout', default=0.1,
                   type=float, help='Non-markovian relation dropout')
group.add_argument('--non_markovian_relation_mlp', action='store_true', default=True,
                   help='Non-markovian relation MLP')
group.add_argument('--non_markovian_relation_activation', default='gelu', type=str,
                   help='Non-markovian relation activation function')
group.add_argument('--non_markovian_relation_num_layers', default=1,
                   type=int, help="Non-markovian relation number of layers")
group.add_argument('--non_markovian_relation_intermediate_size', default=3072,
                   type=int, help="Non-markovian relation intermediate size")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--loss', default="soft_labels", help='Loss function', choices=['XE', 'HXE', 'aa_loss_add', 'aa_loss_only', 'soft_labels'])
group.add_argument('--hxe-alpha', default=0.2, type=float, help='Hierarchical cross entropy alpha')
group.add_argument('--soft-labels-beta', default=15, type=float, help='Soft labels beta')
group.add_argument('--hierarchy', default="random", type=str, help='Hierarchy')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='Number of steps to accumulate gradients across per iteration')
group.add_argument('--opt-kwargs', nargs='*',
                   default={}, action=utils.ParseKwargs)


# Distillation parameters
group = parser.add_argument_group('Distillation parameters')
group.add_argument('--distill', action='store_true', default=False,
                   help='Enable distillation from teacher')
group.add_argument('--distill-mode', type=str, default='all_tokens', metavar='MODE',
                   help='Distillation mode')
group.add_argument('--teacher-path', type=str, default='', metavar='PATH',
                   help='Path to teacher model checkpoint')
group.add_argument('--distill-teacher', type=str, default='mamba', metavar='MODEL',
                   help='Teacher model architecture')
group.add_argument('--distill-temperature', type=float, default=2.0,   
                     help='distillation temperature')
group.add_argument('--distill-alpha', type=float, default=0.5,
                     help='distillation alpha')
group.add_argument('--distill-type', type=str, default='soft',
                    choices=['none', 'soft', 'hard', 'reverse_soft'],
                     help='distillation type (soft, hard)')
group.add_argument('--distill-tie-weights', action='store_true', default=False, help='Tie embedding layer and distill head weights')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--early-stop', action='store_true', default=False, help='Enable early stopping')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-8, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Model Exponential Moving Average
group = parser.add_argument_group(
    'Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--resume', default='', type=str, metavar='PATH', nargs='?',
                   help='resume from last checkpoint')
group.add_argument('--accelerator', default='cuda', type=str, metavar='ACC', choices=['cpu', 'cuda', 'tpu', 'mps'],
                   help='Accelerator device')
group.add_argument('--num_workers', default=4, type=int,
                   metavar='N', help='Number of data loader workers')

group.add_argument('--distributed-backend', default='ddp', type=str, metavar='BACKEND',
                   choices=['ddp', 'ddp2', 'ddp_spawn', 'ddp_sharded', 'horovod'], help='Distributed backend')
group.add_argument('--sync-bn', action='store_true', default=False,
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')

group.add_argument('--num_nodes', type=int, default=1,
                   metavar='N', help='Number of nodes')
group.add_argument('--num_devices', type=int, default=1,
                   metavar='N', help='Number of devices')

group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--checkpoint-hist', type=int, default=2, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('--val-metric-mode', type=str, default='min', metavar='VAL_METRIC_MODE',
                   help='Best metric mode (default: "min"')

group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--amp', action='store_true', default=False,
                   help='use mixed precision training')
group.add_argument('--log-interval', type=int, default=1, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--pin-mem', action='store_true', default=True,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='./', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--eval-metric', default='val_loss', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tree = None
    if not args.distill:
        args.distill_mode = None
        args.distill_type = None

    aa_loss = None
    soft_labels = None
    with open(args.output + 'training_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.mode == "mlm":
        args.bidirectional = True
    if args.mode == "clm":
        args.bidirectional = False

    if args.loss == "XE":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == "HXE":
        assert args.hxe_alpha is not None, "Alpha value must be provided for HXE loss."
        assert args.tokenizer == "3mer", "HXE loss can only be used with 3mer tokenizer."
        assert args.data_type == "nuc", "HXE loss can only be used with nucleotide data."
        from hxe import HierarchicalCrossEntropyLoss
        from create_tree import all_tokens,  get_weighting, get_classes, shuffle_and_merge_codons
        if args.hierarchy == "random":
            tree = shuffle_and_merge_codons(all_tokens)
            weights = get_weighting(tree, "exponential", value=args.hxe_alpha)
            classes = get_classes(tree)[0]
            criterion = HierarchicalCrossEntropyLoss(tree, classes, weights)
        else:
            weights = get_weighting(all_tokens, "exponential", value=args.hxe_alpha)
            classes = get_classes(all_tokens)[0]
            criterion = HierarchicalCrossEntropyLoss(all_tokens, classes, weights)
        tree = classes

    elif "aa_loss" in args.loss:
        assert args.data_type == "nuc", "AA loss can only be used with nucleotide data."
        assert args.tokenizer == "3mer", "AA loss loss can only be used with 3mer tokenizer."
        criterion = torch.nn.CrossEntropyLoss()
        aa_loss = "add" if "add" in args.loss else "only"

    elif args.loss == "soft_labels":
        assert args.tokenizer == "3mer", "Soft labels can only be used with 3mer tokenizer."
        assert args.data_type == "nuc", "Soft labels can only be used with nucleotide data."
        assert args.soft_labels_beta is not None, "Beta value must be provided for soft labels."
        from soft_label_utils import distance_dict, classes, make_all_soft_labels
        soft_labels = make_all_soft_labels(distance_dict, classes, args.soft_labels_beta)
        criterion = torch.nn.KLDivLoss()

    if args.amp:
        criterion = criterion.to(dtype=torch.bfloat16)
    # else:
    
    if args.data_type == "aa":
        from utils import encode_sequence_aa
        from utils import AA_TO_ID, ID_TO_AA
        tokenizer = encode_sequence_aa
        args.vocab_size = len(AA_TO_ID)
        CODON_TO_ID = AA_TO_ID
        ID_TO_CODON = ID_TO_AA
    
    elif args.data_type == "nuc":
        if args.tokenizer == "3mer":
            from utils import CODON_TO_ID_3, ID_TO_CODON_3
            tokenizer = partial(encode_sequence_codon, overlap=args.overlap, k=3, tree=tree)
            CODON_TO_ID = CODON_TO_ID_3
            ID_TO_CODON = ID_TO_CODON_3
            args.vocab_size = len(CODON_TO_ID)
            
        elif args.tokenizer == "nuc":
            from utils import CODON_TO_ID_1, ID_TO_CODON_1
            tokenizer = partial(encode_sequence_codon, overlap=args.overlap, k=1)
            CODON_TO_ID = CODON_TO_ID_1
            ID_TO_CODON = ID_TO_CODON_1
            args.vocab_size = len(CODON_TO_ID) + 1

        elif args.tokenizer == "6mer":
            from utils import CODON_TO_ID_6, ID_TO_CODON_6
            tokenizer = partial(encode_sequence_codon, overlap=args.overlap, k=6)
            CODON_TO_ID = CODON_TO_ID_6
            ID_TO_CODON = ID_TO_CODON_6
            args.vocab_size = len(CODON_TO_ID) + 1

        else:
            from utils import train_tokenizer
            from datasets import Features, Value
            from datasets import load_dataset
            import os

            ft = Features({args.column_name: Value('string')})
            file = os.path.join(args.data_dir, args.train_file)
            dataset = load_dataset(args.data_type, data_files=file, skiprows=1, features=ft)
            if args.tokenizer == "bpe":
                tokenizer = train_tokenizer(dataset, args.column_name, "bpe", args.vocab_size)
            elif args.tokenizer == "wp":
                tokenizer = train_tokenizer(dataset, args.column_name, "wp", args.vocab_size)
            elif args.tokenizer == "ug":
                tokenizer = train_tokenizer(dataset, args.column_name, "ug", args.vocab_size)
            else:
                raise ValueError("Tokenizer must be one of 'bpe', 'wp', or 'ug'.")
            
            print("Tokenizer trained successfully.")
            CODON_TO_ID = tokenizer.get_vocab()
            tokenizer.save_pretrained(args.output)

    if args.model_name == "mamba_hf":
        from models.mamba_hf import MambaConfig, MambaConfigNew, Mamba
        token_to_id=AA_TO_ID if args.data_type == "aa" else CODON_TO_ID
        model_config = MambaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            state_size=16,
            num_hidden_layers=args.num_layers,
            layer_norm_epsilon=1e-5,
            pad_token_id=token_to_id["<pad>"],
            bos_token_id=token_to_id['<cls>'],
            eos_token_id=token_to_id['<eos>'],
            expand=args.intermediate_size // args.hidden_size,
            conv_kernel=4,
            use_bias=False,
            use_conv_bias=True,
            hidden_act="silu",
            initializer_range=0.1,
            residual_in_fp32=True,
            time_step_rank="auto",
            time_step_scale=1.0,
            time_step_min=0.001,
            time_step_max=0.1,
            time_step_init_scheme="random",
            time_step_floor=1e-4,
            rescale_prenorm_residual=False,
            use_cache=True,
        )

        model_config = MambaConfigNew(
        mamba_config=model_config,
        tie_embeddings=args.tie_embeddings,
        pos_embedding=args.pos_embedding,
        pos_embedding_apply=args.pos_embedding_apply,
        pos_embedding_dim=args.pos_embedding_dim,
        distillation_mode=args.distill_mode,
        distillation=args.distill,
        token_to_id=token_to_id
        )
        model = Mamba(config=model_config)
    
    elif args.model_name == "mamba":
        from models.mamba import MambaConfig, MambaLMHeadModel
        model_config = MambaConfig()
        model_config.bidirectional = args.bidirectional
        model_config.vocab_size = args.vocab_size
        model_config.n_layer = args.num_layers
        model_config.pad_vocab_size_multiple = args.pad_vocab_size_multiple
        model_config.tie_embeddings = args.tie_embeddings
        model_config.d_model = args.hidden_size
        model_config.d_intermediate = args.intermediate_size
        model_config.distillation = args.distill
        model_config.pos_embedding = args.pos_embedding
        model_config.pos_embedding_apply = args.pos_embedding_apply
        model_config.pos_embedding_dim = args.pos_embedding_dim
        model_config.distillation_mode = args.distill_mode
        model_config.distillation_tie_weights = args.distill_tie_weights
        model_config.max_seq_len=args.max_seq_len
        model_config.non_markovian_relation = args.non_markovian_relation
        model_config.non_markovian_relation_mlp = args.non_markovian_relation_mlp
        model_config.non_markovian_relation_cfg = {
            "num_heads": args.non_markovian_relation_num_heads,
            "kernel_size": args.non_markovian_relation_kernel_size,
        }
        model = MambaLMHeadModel(config=model_config)
        
    elif args.model_name == "hyena":
        from models.hyena import HyenaConfig, HyenaConfigNew, HyenaDNA
        model_config = HyenaConfig(
            vocab_size=args.vocab_size,
            d_model=args.hidden_size,
            d_inner=args.intermediate_size,
            use_bias=True,
            train_freq=True,
            max_seq_len=args.max_seq_len,
            n_layer=args.num_layers,
            num_inner_mlps=2,
            hyena_order=2,
            short_filter_order=3,
            filter_order=64,
            activation_freq=1,
            embed_dropout=args.dropout,
            bidirectional=args.bidirectional,
            hyena_dropout=args.attn_dropout,
            hyena_filter_dropout=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            pad_vocab_size_multiple=args.pad_vocab_size_multiple,
        )
        model_config = HyenaConfigNew(
            hyena_config=model_config,
            tie_embeddings=args.tie_embeddings,
            pos_embedding=args.pos_embedding,
            pos_embedding_apply=args.pos_embedding_apply,
            pos_embedding_dim=args.pos_embedding_dim,
            distillation_mode=args.distill_mode,
            distillation=args.distill,
            token_to_id=AA_TO_ID if args.data_type == "aa" else CODON_TO_ID,
        )
        model = HyenaDNA(config=model_config)
    elif args.model_name in ["bert", "gpt2"]:
        from models.transformer import TransformerConfig, Transformer
        model_config = TransformerConfig()
        model_config.model = args.model_name
        model_config.amp = args.amp
        model_config.token_to_id = AA_TO_ID if args.data_type == "aa" else CODON_TO_ID
        model_config.bidirectional = args.bidirectional
        model_config.vocab_size = args.vocab_size
        model_config.d_model = args.hidden_size
        model_config.d_intermediate = args.intermediate_size
        model_config.n_layer = args.num_layers
        model_config.pad_vocab_size_multiple = args.pad_vocab_size_multiple
        model_config.distillation = args.distill
        model_config.pos_embedding = args.pos_embedding
        model_config.pos_embedding_apply = args.pos_embedding_apply
        model_config.pos_embedding_dim = args.pos_embedding_dim
        model_config.distillation_mode = args.distill_mode
        model_config.num_heads = args.num_heads
        model_config.attn_dropout = args.attn_dropout
        model_config.tree = tree
        model_config.attn_layer_idx = [i for i in range(args.num_layers)]
        model_config.tie_embeddings = args.tie_embeddings
        model = Transformer(config=model_config)

   

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    data_train = DataAntiBody(
        args.data_dir,
        args.train_file,
        tokenizer,
        CODON_TO_ID,
        args.position_ids,
        args.masking_strategy,
        args.masking_ratio,
        args.max_seq_len,
        args.mode,
        extension=args.file_extension,
        column_name=args.column_name,
        tree=tree,
        num_samples=args.num_samples
    )
    if args.val_file is None and args.test_file is None:
        data_train, data_val, data_test = torch.utils.data.random_split(data_train, [int(0.8 * len(data_train)), int(
            0.1 * len(data_train)), len(data_train) - int(0.8 * len(data_train)) - int(0.1 * len(data_train))])
    if args.val_file is None:
        data_val = data_test
    if args.test_file is None:
        data_test = data_val
    else:
        data_val = DataAntiBody(
            args.data_dir,
            args.val_file,
            tokenizer,
            CODON_TO_ID,
            args.position_ids,
            args.masking_strategy,
            args.masking_ratio,
            args.max_seq_len,
            args.mode,
            tree=tree,
        )
        data_test = DataAntiBody(
            args.data_dir,
            args.test_file,
            tokenizer,
            CODON_TO_ID,
            args.position_ids,
            args.masking_strategy,
            args.masking_ratio,
            args.max_seq_len,
            args.mode,
            tree=tree,
        )

    loader_val = DataLoader(
        data_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    loader_test = DataLoader(
        data_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    loader_train = DataLoader(
        data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    updates_per_epoch = (len(loader_train) +
                         args.grad_accum_steps - 1) // args.grad_accum_steps
   
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    label_metrics = MetricCollection(
        [Accuracy(task="multiclass", num_classes=model.vocab_size)])
    logits_metrics = MetricCollection([Perplexity()])
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor=args.eval_metric,
        dirpath=args.output,
        filename=args.train_file + "-" + args.model_name + '-{epoch:02d},{val_loss:.2f}',
        save_top_k=args.checkpoint_hist,
        mode=args.val_metric_mode,
        save_last=True,
        auto_insert_metric_name=True)
    callbacks.append(checkpoint_callback)
    if args.early_stop:
        early_stop_callback = EarlyStopping(
            monitor=args.eval_metric,
            patience=args.patience_epochs,
            mode=args.val_metric_mode,
        )
        callbacks.append(early_stop_callback)
    
    if not args.distill:
        teacher = None
    
    else:
        pass
    
    class LitProgressBar(ProgressBar):

        def __init__(self):
            super().__init__()  # don't forget this :)
            self.enable = True

        def disable(self):
            self.enable = False

        def get_metrics(self, trainer, model):
            # don't show the version number
            items = super().get_metrics(trainer, model)
            items.pop("v_num", None)
            return items

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
            metrics = self.get_metrics(trainer, pl_module)
            sys.stdout.flush()
            sys.stdout.write(f'[{batch_idx}/{self.total_train_batches}, {metrics}] \r')

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.output + "/logs")
    model_l = AntibodyLLMLightening(
        model,
        args.mode,
        args.distill,
        args.distill_type,
        args.distill_alpha,
        args.distill_temperature,
        teacher,
        criterion,
        optimizer,
        lr_scheduler,
        label_metrics,
        logits_metrics,
        tree=tree,
        token_to_id=CODON_TO_ID,
        id_to_token=ID_TO_CODON,
        aa_loss=aa_loss,
        soft_labels=soft_labels
    )
    trainer = L.Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        precision="bf16-mixed" if args.amp else 32,
        accumulate_grad_batches=args.grad_accum_steps,
        gradient_clip_val=args.clip_grad,
        gradient_clip_algorithm=args.clip_mode,
        num_nodes=args.num_nodes,
        devices=args.num_devices,
        strategy=args.distributed_backend,
        sync_batchnorm=args.sync_bn,
        log_every_n_steps=args.log_interval,
    )
    trainer.fit(model_l, loader_train, loader_val)

if __name__ == "__main__":
    main()
