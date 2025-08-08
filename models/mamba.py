import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from .pos_embed import PositionalEmbeddingSinCos
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from dataclasses import dataclass, field

@dataclass
class MambaConfig:
    ssm_layer: str = "Mamba"
    d_model: int = 512
    d_intermediate: int = 0
    n_layer: int = 12
    vocab_size: int = 38,
    pos_embedding: str = "absolute"
    pos_embedding_apply: str = "add"
    pos_embedding_dim: int = None
    non_markovian_relation: bool = False
    non_markovian_relation_mlp: bool = False
    non_markovian_relation_cfg: dict = field(default_factory=dict),
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = False
    pad_vocab_size_multiple: int = 1
    max_seq_len: int = 2048
    tie_embeddings: bool = False
    bidirectional: bool = False
    distillation: bool = False
    distillation_mode: str = "same"
    distillation_tie_weights: bool = False


# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor

from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class Block(nn.Module):
    def __init__(
        self, dim, non_markovian_cls, non_markovian_mlp, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.non_markovian_norm = norm_cls(dim)
        self.non_markovian = non_markovian_cls(dim) if non_markovian_cls is not None else nn.Identity()
        self.non_markovian_mlp = mlp_cls(dim) if (non_markovian_mlp and self.non_markovian is not nn.Identity) else nn.Identity()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.non_markovian(hidden_states)
        hidden_states = self.non_markovian_norm(hidden_states)
        hidden_states = self.non_markovian_mlp(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        d_intermediate,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        non_markovian_relation=False,
        non_markovian_relation_cfg=None,
        non_markovian_mlp=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if non_markovian_relation:
        non_markovian_cls = nn.Identity()
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        non_markovian_cls if non_markovian_relation else None,
        non_markovian_mlp,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            d_intermediate: int,
            vocab_size: int,
            ssm_layer: str = "Mamba",
            pos_embedding: str = None,
            pos_embedding_apply: str = None,
            pos_embedding_dim: int = None,
            max_len: int = 2048,
            ssm_cfg =  None,
            attn_layer_idx=None,
            attn_cfg=None,
            non_markovian_relation=False,
            non_markovian_relation_cfg=None,
            non_markovian_mlp=False,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        ssm_cfg = {"layer": ssm_layer} if ssm_cfg is None else ssm_cfg
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.pos_embedding_apply = pos_embedding_apply
        add_to_dim = 0
        if pos_embedding is not None:
            assert pos_embedding in ["absolute", "Rotary", "SinCos"] and max_len is not None, "Invalid pos_embedding"
            assert pos_embedding_apply in ["concat", "add"], "Invalid pos_embedding_apply"

            if pos_embedding_apply == "concat":
                add_to_dim = pos_embedding_dim
                assert pos_embedding_dim is not None, "pos_embedding_dim must be provided for concat"

        if pos_embedding == "absolute":
            if pos_embedding_apply == "concat":
                self.embedding_pos = nn.Embedding(max_len, pos_embedding_dim, **factory_kwargs)
            elif pos_embedding_apply == "add":
                self.embedding_pos = nn.Embedding(max_len, d_model, **factory_kwargs)

        elif pos_embedding == "Rotary":
            raise NotImplementedError("Rotary Positional Embedding is not implemented yet")

        elif pos_embedding == "SinCos":
            if pos_embedding_apply == "concat":
                self.embedding_pos = PositionalEmbeddingSinCos(pos_embedding_dim, max_len=max_len)
            elif pos_embedding_apply == "add":
                self.embedding_pos = PositionalEmbeddingSinCos(d_model, max_len=max_len)

        else:
            self.embedding_pos = None
        self.device = device
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model + add_to_dim,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    non_markovian_relation=non_markovian_relation,
                    non_markovian_relation_cfg=non_markovian_relation_cfg,
                    non_markovian_mlp=non_markovian_mlp,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        if self.embedding_pos is not None:
            if self.pos_embedding_apply == "concat":
                hidden_states = torch.cat([hidden_states, self.embedding_pos(position_ids).to(hidden_states.device)], dim=-1)
            elif self.pos_embedding_apply == "add":
                hidden_states = hidden_states + self.embedding_pos(position_ids).to(hidden_states.device)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class BiDirectionMixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            d_intermediate: int,
            vocab_size: int,
            ssm_layer: str = "Mamba",
            pos_embedding: str = None,
            pos_embedding_apply: str = None,
            pos_embedding_dim: int = None,
            max_len: int = 2048,
            ssm_cfg=None,
            attn_layer_idx=None,
            attn_cfg=None,
            non_markovian_relation=False,
            non_markovian_relation_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        ssm_cfg = {"layer": ssm_layer} if ssm_cfg is None else ssm_cfg
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.gate = nn.Linear(2 * d_model, 1, )
        self.pos_embedding_apply = pos_embedding_apply
        add_to_dim = 0
        if pos_embedding is not None:
            assert pos_embedding in ["absolute", "Rotary", "SinCos"] and max_len is not None, "Invalid pos_embedding"
            assert pos_embedding_apply in ["concat", "add"], "Invalid pos_embedding_apply"

            if pos_embedding_apply == "concat":
                add_to_dim = pos_embedding_dim
                assert pos_embedding_dim is not None, "pos_embedding_dim must be provided for concat"

        if pos_embedding == "absolute":
            if pos_embedding_apply == "concat":
                self.embedding_pos = nn.Embedding(max_len, pos_embedding_dim, **factory_kwargs)
            elif pos_embedding_apply == "add":
                self.embedding_pos = nn.Embedding(max_len, d_model, **factory_kwargs)

        elif pos_embedding == "Rotary":
            raise NotImplementedError("Rotary Positional Embedding is not implemented yet")

        elif pos_embedding == "SinCos":
            if pos_embedding_apply == "concat":
                self.embedding_pos = PositionalEmbeddingSinCos(pos_embedding_dim, max_len=max_len)
            elif pos_embedding_apply == "add":
                self.embedding_pos = PositionalEmbeddingSinCos(d_model, max_len=max_len)

        else:
            self.embedding_pos = None
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.forward_layers = nn.ModuleList(
            [
                create_block(
                    d_model + add_to_dim,
                    ssm_cfg=ssm_cfg,
                    d_intermediate=d_intermediate,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    non_markovian_relation=non_markovian_relation,
                    non_markovian_relation_cfg=non_markovian_relation_cfg,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.backward_layers = nn.ModuleList(
            [
                create_block(
                    d_model + add_to_dim,
                    ssm_cfg=ssm_cfg,
                    d_intermediate=d_intermediate,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    non_markovian_relation=non_markovian_relation,
                    non_markovian_relation_cfg=non_markovian_relation_cfg,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        self.hidden_fc = nn.ModuleList(
            [nn.Linear(2 * d_model, d_model) for i in range(n_layer)]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, position_ids, embedding=None, inference_params=None):
        hidden_states = self.embedding(input_ids)
        embedding = torch.zeros_like(hidden_states) if embedding is None else embedding
        if self.embedding_pos is not None:
            if self.pos_embedding_apply == "concat":
                hidden_states = torch.cat([hidden_states, self.embedding_pos(position_ids).to(hidden_states.device)], dim=-1)
            elif self.pos_embedding_apply == "add":
                hidden_states = hidden_states + self.embedding_pos(position_ids).to(hidden_states.device)
        gate = self.gate(torch.cat([hidden_states, embedding], dim=-1)).sigmoid()
        hidden_states = hidden_states * gate + embedding * (1 - gate)
        residual = None
        for f_layer, b_layer, h_fc in zip(
                self.forward_layers, self.backward_layers, self.hidden_fc
        ):
            hidden_states_f, residual_f = f_layer(
                hidden_states, residual, inference_params=inference_params
            )
            flip_residual = residual.flip([1]) if residual is not None else None
            hidden_states_b, residual_b = b_layer(
                hidden_states.flip([1]), flip_residual, inference_params=inference_params
            )
            hidden_states = h_fc(torch.cat([hidden_states_f, hidden_states_b.flip([1])], dim=-1))
            residual = 0.5 * (residual_f + residual_b.flip([1]))

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
            self,
            config: MambaConfig,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        bidirectional = config.bidirectional
        ssm_layer = config.ssm_layer
        pos_embedding = config.pos_embedding
        pos_embedding_apply = config.pos_embedding_apply
        pos_embedding_dim = config.pos_embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.vocab_size = vocab_size
        self.backbone = partial(BiDirectionMixerModel if bidirectional else MixerModel,
                                d_model=d_model,
                                n_layer=n_layer,
                                d_intermediate=d_intermediate,
                                vocab_size=vocab_size,
                                ssm_layer=ssm_layer,
                                ssm_cfg=ssm_cfg,
                                attn_layer_idx=attn_layer_idx,
                                max_len=config.max_seq_len,
                                attn_cfg=attn_cfg,
                                pos_embedding=pos_embedding,
                                pos_embedding_apply=pos_embedding_apply,
                                pos_embedding_dim=pos_embedding_dim,
                                non_markovian_relation=config.non_markovian_relation,
                                non_markovian_relation_cfg=config.non_markovian_relation_cfg,
                                rms_norm=rms_norm,
                                initializer_cfg=initializer_cfg,
                                fused_add_norm=fused_add_norm,
                                residual_in_fp32=residual_in_fp32,
                                **factory_kwargs,
                                )()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        if config.distillation:
            self.distill_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()
        self.tie_weights_distill()
   
    def tie_weights_distill(self):
        if self.config.distillation_tie_weights and self.config.distillation:
            self.distill_head.weight = self.backbone.embeddings.word_embeddings.weight

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, masked_tokens=None, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, position_ids, inference_params=inference_params, **mixer_kwargs)
        distill_logits = None
        if self.config.distillation:
            if self.config.distillation_mode == "all_tokens":
                hidden_states_distill = hidden_states[:, 1::2, :]
                hidden_states = hidden_states[:, ::2, :]
            elif self.config.distillation_mode == "last_token":
                hidden_states_distill = hidden_states[:, -1, :]
                hidden_states = hidden_states[:, :-1, :]
            elif self.config.distillation_mode == "same":
                hidden_states_distill = hidden_states
            else:
                raise ValueError(
                    f"distillation_mode {self.config.distillation_mode} not recognized"
                )
            distill_logits = self.distill_head(hidden_states_distill)
        
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if masked_tokens is not None:
            lm_logits = self.lm_head(hidden_states[masked_tokens])
        else:
            lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "hidden_states", "distill_logits"])
        return CausalLMOutput(logits=lm_logits, distill_logits=distill_logits, hidden_states=hidden_states)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)