from transformers.models.mamba import MambaConfig, MambaModel
from collections import namedtuple
from dataclasses import dataclass, field
from torch import nn

@dataclass
class MambaConfigNew:
    mamba_config: MambaConfig
    tie_embeddings: bool = True
    pos_embedding: str = "learned"
    pos_embedding_apply: str = "add"
    pos_embedding_dim: int = None
    distillation: bool = False
    distillation_mode: str = "all_tokens"
    token_to_id: dict = field(default_factory=dict)
    

class Mamba(nn.Module):
    def __init__(self, config: MambaConfigNew) -> None:
        super().__init__()
        self.vocab_size = config.mamba_config.vocab_size
        self.distillation_mode = config.distillation_mode
        self.distillation = config.distillation
        self.token_to_id = config.token_to_id
        self.model = MambaModel(config.mamba_config)
        self.lm_head = nn.Linear(config.mamba_config.hidden_size, self.vocab_size, bias=False)
        if self.distillation:
            self.distill_head = nn.Linear(
                config.mamba_config.hidden_size, self.vocab_size, bias=False)

        if config.tie_embeddings:
            self._tie_embeddings()

    def _tie_embeddings(self):
        self.lm_head.weight = self.model.embeddings.weight

    def forward(self, x, pos_id, masked_tokens=None):
        attention_mask = (x != self.token_to_id["<pad>"]).to(x.dtype)
        x = self.model(x)
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
            lm_logits = self.lm_head(x[0][masked_tokens])
        else:
            lm_logits = self.lm_head(x[0])
        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "hidden_states", "distill_logits"])
        return CausalLMOutput(logits=lm_logits, distill_logits=distill_logits, hidden_states=x)
