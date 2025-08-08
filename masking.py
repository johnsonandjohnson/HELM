from utils import AA_TO_ID, sample_lengths, encode_sequence_aa
import numpy as np


def single_token_masking(ids, pos_ids, mask_ratio, token_to_id, tree=None):
    """Mask a sequence of tokens."""
    assert 0 <= mask_ratio <= 1, "Mask ratio must be between 0 and 1."
    cls_token = token_to_id["<cls>"] if tree is None else tree.index("<cls>")
    eos_token = token_to_id["<eos>"] if tree is None else tree.index("<eos>")
    pad_token = token_to_id["<pad>"] if tree is None else tree.index("<pad>")
    mask_token = token_to_id["<mask>"] if tree is None else tree.index("<mask>")
    ids = np.array(ids)
    noise = np.random.rand(ids.shape[0])
    special_tokens = (ids == cls_token) | (ids == eos_token) | (ids == pad_token) | (ids == mask_token)
    mask = (noise < np.array([mask_ratio])) & (~special_tokens)
    ids[mask] = mask_token
    return ids, pos_ids


def multi_token_masking(ids, pos_ids, max_masked, mask_ratio, token_to_id):
    """Mask multiple tokens consecutively in  the sequence"""
    assert max_masked > 0, "Number of masked tokens must be greater than 0."
    token_to_id = token_to_id
    special_token_pos = (ids == token_to_id["<cls>"]) | (ids == token_to_id['<eos>']) | (ids == token_to_id['<pad>']) | (ids == token_to_id['<mask>'])
    sep_seq = np.split(ids, np.where(special_token_pos)[0])
    sep_pos = np.split(pos_ids, np.where(special_token_pos)[0])
    max_len_index = np.argmax([len(s) for s in sep_seq])
    sep_pos_new = sep_pos[max_len_index]
    start_patches = np.random.choice(np.arange(sep_pos_new[0] + 1, sep_pos_new[-1]), max_masked, replace=False)
    for i in range(len(start_patches)):
        if max_masked == 1:
            s_to_be_masked_len = len(sep_seq[max_len_index])
        else:
            if i == len(start_patches):
                break
            s_to_be_masked_len = start_patches[i + 1] - start_patches[i]
        patch_len = sample_lengths(s_to_be_masked_len, mask_ratio)
        seq1 = ids[:start_patches[i]]
        seq2 = ids[start_patches[i]:start_patches[i] + patch_len]
        pos_id1 = pos_ids[:start_patches[i]]
        pos_id2 = pos_ids[start_patches[i]:start_patches[i] + patch_len]
        seq3 = ids[start_patches[i] + patch_len:]
        pos_id3 = pos_ids[start_patches[i] + patch_len:]
        ids = np.concatenate([seq1, [token_to_id[f"<mask_{i + 1}>"]], seq3, [token_to_id[f"<mask_{i + 1}>"]], seq2])
        pos_ids = np.concatenate([pos_id1, [pos_id2[0]], pos_id3, [pos_id2[0]], pos_id2])

    return ids, pos_ids

