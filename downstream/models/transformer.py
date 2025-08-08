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
from .hf_models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel
from transformers.generation import GenerationMixin
from transformers import GPT2PreTrainedModel, PreTrainedModel, GenerationConfig
def setup_generation_config(model_config, pad_id, eos_id, cls_id, **kwargs):
    # Start with default generation config
    generation_config = GenerationConfig.from_model_config(model_config)
    generation_config.pad_token_id = pad_id
    generation_config.bos_token_id = cls_id
    generation_config.eos_token_id = eos_id
    generation_config.max_length = 250
    generation_config.do_sample = True
    generation_config.top_k = 10
    generation_config.top_p = 0.95
    generation_config.forced_eos_token_id = generation_config.eos_token_id
    generation_config.do_sample = kwargs.get('do_sample', True)
    generation_config.num_beams = kwargs.get('num_beams', 1)
    generation_config.temperature = kwargs.get('temperature', 1.0)
    generation_config.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
    generation_config.length_penalty = kwargs.get('length_penalty', 1.0)
    generation_config.no_repeat_ngram_size = kwargs.get('no_repeat_ngram_size', 5)
    # Update with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(generation_config, key):
            setattr(generation_config, key, value)
        else:
            print(f"Warning: '{key}' is not a valid GenerationConfig parameter. Ignoring.")
    
    return generation_config
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
    tie_embeddings: bool = True
    bidirectional: bool = False
    pos_embedding: str = "absolute"
    pos_embedding_apply: str = "add"
    pos_embedding_dim: int = None
    distillation_mode = None
    distillation = False 

    def read_json(self, config):
        for k, v in config.items():
            setattr(self, k, v)
        return self


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
        model_conf = GPT2Config(
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
            bos_token_id=config.token_to_id["<cls>"],
            eos_token_id=config.token_to_id["<eos>"],
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            is_decoder=not config.bidirectional,
            position_embedding_apply=config.pos_embedding_apply,
            position_embedding_type=config.pos_embedding,
        )
    return model_conf

SUPPORTED_MODELS = {
    "bert": BertModel,
    "gpt2": GPT2Model,
}


class Transformer(GPT2PreTrainedModel):
    def __init__(self, config: TransformerConfig) -> None:
        model_config = fill_config(config.model, config)
        super().__init__(model_config)
        self.model_type = config.model
        self.vocab_size = config.vocab_size
        self.distillation_mode = config.distillation_mode
        self.distillation = config.distillation
        self.token_to_id = config.token_to_id
        model = SUPPORTED_MODELS[config.model]
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

    
    def forward(self, input_ids, position_ids=None, masked_tokens=None, inference_params=None, num_last_tokens=1, past_key_values=None, use_cache=False, attention_mask=None, token_type_ids=None, return_dict=True, output_attentions=False, output_hidden_states=False, **kwargs):
        x = input_ids
        attention_mask = (x != self.token_to_id["<pad>"]).to(x.dtype) if attention_mask is None else attention_mask
        if position_ids is None:
            position_ids = torch.arange(x.size(1), device=x.device)
            position_ids = position_ids.unsqueeze(0).expand_as(x)
        
        x = self.model(x, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, token_type_ids=token_type_ids, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, **kwargs)
        hidden_states = x.last_hidden_state
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
        
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        if masked_tokens is not None:
            lm_logits = self.lm_head(x.last_hidden_state[masked_tokens])
        else:
            lm_logits = self.lm_head(x.last_hidden_state)
        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "hidden_states", "distill_logits"])
            
        return CausalLMOutput(logits=lm_logits, distill_logits=distill_logits, hidden_states=hidden_states)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs
    from transformers import GenerationConfig


    # Usage in your generate method or wherever you're setting up generation
    def generate(self, input_ids, **kwargs):
        model_config = self.config  # Assuming this is your GPT2Config
        generation_config = setup_generation_config(model_config, **kwargs)
        
        # Now use this generation_config for your generation process
        outputs = super().generate(
            input_ids,
            generation_config=generation_config,
        )
        
        return outputs