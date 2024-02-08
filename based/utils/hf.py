""" Source: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/hf.py """

import json
import torch
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


def load_config_hf(model_name):
    # resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    # return json.load(open(resolved_archive_file))

    # SA: replace this when done with dev work
    config_check = {
        "n_embd": 1792,
        "n_inner": 3584,
        "n_head": 16,
        "n_layer": 36,

        "activation_function": "swiglu",
        "resid_pdrop": 0.0,
        "residual_in_fp32": True,
        "pad_vocab_size_multiple": 8,
        "use_flash_attn": True,
        "special_initializer": True,
        "max_position_embeddings": 0,

        "alt_mixer_layers": [1, 6, 11, 16, 21, 27, 33],
        "alt_2_mixer_layers": [2, 7, 12, 17, 22, 28, 34],
        "mixer": {
            "_target_": "based.models.mixers.base_conv.BaseConvWithSiLU4",
            "expand_proj": 4,
            "l_max": 2048,
            "kernel_sizes": [3],
        },
        "alt_mixer": {
            "_target_": "based.models.mixers.linear_attn.LinearAttention",
            "feature_dim": 16,
            "l_max": 2048,
            "num_heads": 16,
            "num_key_value_heads": 16,
            "train_view": "linear",
        },
        "alt_mixer_2": {
            "_target_": "based.models.mixers.slide_fa2.SlidingsMHA",
            "causal": True,
            "num_heads": 16,
            "window_size": 128,
        }
    }
    return config_check


def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
