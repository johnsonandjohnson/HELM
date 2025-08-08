# Copyright (c) 2023, Tri Dao, Dan Fu.
# Simplified, mostly standalone version of LongConvLM for synthetics.

import math
from functools import partial

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from dataclasses import dataclass, field
from .hf_models.bert import BertConfig, BertModel
from .hf_models.gpt2 import GPT2Config, GPT2Model


class BertConfigNew(BertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position_embedding_apply = kwargs.get(
            "pos_embedding_apply", "add")
        self.position_embedding_dim = kwargs.get("pos_embedding_dim", None)


class GPT2ConfigNew(GPT2Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.amp = kwargs.get("amp", False)
        self.position_embedding_apply = kwargs.get(
            "pos_embedding_apply", "add")
        self.position_embedding_dim = kwargs.get("pos_embedding_dim", None)

@dataclass
class TransformerConfig:
    model: str = "bert"
    amp: bool = False
    d_model: int = 512
    d_intermediate: int = 0
    token_to_id: dict = field(default_factory=dict)
    n_layer: int = 12
    vocab_size: int = 38,
    max_position_embeddings: int = 1024
    resid_dropout: float = 0.0
    embed_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    attn_layer_idx: list = field(default_factory=list)
    num_heads: int = 8
    attn_dropout: float = 0.0
    rms_norm: bool = True
    residual_in_fp32: bool = True
    amp: bool = False
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False
    bidirectional: bool = False
    pos_embedding: str = "learned"
    pos_embedding_apply: str = "add"
    pos_embedding_dim: int = None
    distillation_mode = None
    distillation = False 
    tree = None
      


def fill_config(model_name, config: TransformerConfig, **kwargs):
    if model_name == "bert":
        model_conf = BertConfigNew(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=config.n_layer,
            num_attention_heads=config.num_heads,
            intermediate_size=config.d_intermediate,
            hidden_act="gelu",
            hidden_dropout_prob=config.embed_dropout,
            attention_probs_dropout_prob=config.attn_dropout,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=1,
            layer_norm_eps=config.layer_norm_epsilon,
            pad_token_id=1,
            position_embedding_type=config.pos_embedding,
            use_cache=True,
            is_decoder=not config.bidirectional,
            pos_embedding_apply=config.pos_embedding_apply,
        )
    if model_name == "gpt2":
        model_conf = GPT2ConfigNew(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.d_model,
            n_layer=config.n_layer,
            n_head=config.num_heads,
            n_inner=config.d_intermediate,
            activation_function="gelu_new",
            resid_pdrop=0,
            embd_pdrop=config.embed_dropout,
            attn_pdrop=config.attn_dropout,
            layer_norm_epsilon=config.layer_norm_epsilon,
            amp=config.amp,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=config.token_to_id["<cls>"] if config.tree is None else config.tree.index("<cls>"),
            eos_token_id=config.token_to_id["<eos>"] if config.tree is None else config.tree.index("<eos>"),
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            is_decoder=not config.bidirectional,
            pos_embedding_apply=config.pos_embedding_apply,
            position_embedding_type=config.pos_embedding,
        )
    return model_conf

SUPPORTED_MODELS = {
    "bert": BertModel,
    "gpt2": GPT2Model,
}


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.model_type = config.model
        self.vocab_size = config.vocab_size
        self.distillation_mode = config.distillation_mode
        self.distillation = config.distillation
        self.token_to_id = config.token_to_id
        self.pad_token = self.token_to_id["<pad>"] if config.tree is None else config.tree.index("<pad>")
        model = SUPPORTED_MODELS[config.model]
        model_config = fill_config(config.model, config)
        if config.model == "bert":
            self.model = model(model_config, add_pooling_layer=False)
        else:
            self.model = model(model_config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if self.distillation:
            self.distill_head = nn.Linear(
                config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self._tie_embeddings()

    def _tie_embeddings(self):
        if self.model_type in ["bert", 'roberta']:
            self.lm_head.weight = self.model.embeddings.word_embeddings.weight
        elif self.model_type == "gpt2":
            self.lm_head.weight = self.model.wte.weight

    def forward(self, x, pos_id, masked_tokens=None):
        attention_mask = (x != self.token_to_id["<pad>"]).to(x.dtype)
        x = self.model(x, position_ids=pos_id, attention_mask=attention_mask)
        distill_logits = None
        if self.distillation:
            if self.distillation_mode == "all_tokens":
                hidden_states_distill = hidden_states[:, 1::2, :]
                hidden_states = hidden_states[:, ::2, :]
            elif self.distillation_mode == "last_token":
                hidden_states_distill = hidden_states[:, -1, :]
                hidden_states = hidden_states[:, :-1, :]
            elif self.distillation_mode == "same":
                hidden_states_distill = hidden_states
            else:
                raise ValueError(
                    f"distillation_mode {self.config.distillation_mode} not recognized"
                )
            distill_logits = self.distill_head(hidden_states_distill)
        if masked_tokens is not None:
            lm_logits = self.lm_head(x.last_hidden_state[masked_tokens])
        else:
            lm_logits = self.lm_head(x.last_hidden_state)
        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "hidden_states", "distill_logits"])
        return CausalLMOutput(logits=lm_logits, distill_logits=distill_logits, hidden_states=x)
