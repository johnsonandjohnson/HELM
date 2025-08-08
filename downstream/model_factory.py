import json
import torch
from create_tree import get_classes, all_tokens
from models.transformer import TransformerConfig, Transformer
from models.mamba import MambaConfig, MambaLMHeadModel
from models.hyena import HyenaConfig, HyenaDNA, HyenaConfigNew
from utils import CODON_TO_ID_3, ID_TO_CODON_3, CODON_TO_ID_1, ID_TO_CODON_1, CODON_TO_ID_6, ID_TO_CODON_6

tokenizer_dict = {
    "nuc": (CODON_TO_ID_1, ID_TO_CODON_1, len(CODON_TO_ID_1) + 1),
    "3mer": (CODON_TO_ID_3, ID_TO_CODON_3, len(CODON_TO_ID_3) + 1),
    "6mer": (CODON_TO_ID_6, ID_TO_CODON_6, len(CODON_TO_ID_6) + 1)
}
def create_model(config, model_path, generation=False):
    checkpoint = torch.load(model_path)['state_dict']
    for key in list(checkpoint.keys()):
        checkpoint[key[6:]] = checkpoint.pop(key)

    pad_id = tokenizer_dict[config["tokenizer"]][0]["<pad>"]
    cls_token_id = tokenizer_dict[config["tokenizer"]][0]["<cls>"]
    eos_id = tokenizer_dict[config["tokenizer"]][0]["<eos>"]

    if config["helm"]:
        classes = get_classes(all_tokens)[0]
        pad_id = classes.index("<pad>")
        cls_token_id = classes.index("<cls>")
        eos_id = classes.index("<eos>")
        checkpoint.pop("ion.onehot_den")
        checkpoint.pop("ion.onehot_num")
        checkpoint.pop("ion.weights")

    if config["model_type"] == "transformer":
        
        model_config = TransformerConfig().read_json(config["model_config"])
        lora_target_modules = ["wte", "c_attn", "c_fc", "c_proj"]
        model_config.token_to_id = tokenizer_dict[config["tokenizer"]][0]
        model_config.vocab_size = tokenizer_dict[config["tokenizer"]][2]
        
        if config["helm"]:
            model_config.tree = classes

            
        model = Transformer(model_config)
        model.load_state_dict(checkpoint)
        model = model.model if not generation else model

    elif config["model_type"] == "mamba":
        model_config = MambaConfig().read_json(config["model_config"])
        model_config.vocab_size = tokenizer_dict[config["tokenizer"]][2]
        lora_target_modules = ["dt_proj", "fc1", "fc2", "x_proj", "in_proj", "out_proj", "embedding"]
        model = MambaLMHeadModel(model_config)
        
        model.load_state_dict(torch.load(model_path))
        model = model.backbone if not generation else model
        if not generation:
            model.resize_position_embeddings(1024)
    
    elif config["model_type"] == "hyena":
        model_config = HyenaConfig().read_json(config["model_config"])
        model_config.vocab_size = tokenizer_dict[config["tokenizer"]][2]
        model_config = HyenaConfigNew(model_config)

        lora_target_modules = ["fc1", "fc2", "in_proj", "out_proj", "bias", "implicit_filter.linear", "word_embeddings"]
        model = HyenaDNA(model_config)
        
        model.load_state_dict(torch.load(model_path))
        model = model.model if not generation else model
        
    else:
        raise ValueError("Model type not recognized")

    return model, (cls_token_id, pad_id, eos_id), lora_target_modules