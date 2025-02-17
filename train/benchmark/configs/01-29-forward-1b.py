import torch

from config import ModelConfig
use_ray = False
torch.random.manual_seed(0)

configs = []
for batch_size in [4]:
    for PREFILL_SIZE in [8192]:
            
            models = {}

            models["mamba"] = ModelConfig(
                _target_="based.models.mamba.MambaLMHeadModel",
                config={
                    "_target_": "based.models.mamba.MambaConfig",
                    "reorder_and_upcast_attn": False,
                    "scale_attn_by_inverse_layer_idx": True,
                    "n_positions": PREFILL_SIZE,
                    "n_embd": 2048,
                    "n_head": 16,
                    "n_layer": 47,
                    "residual_in_fp32": True,
                    "use_flash_attn": True,
                    "fused_dropout_add_ln": False ,
                    "fused_mlp": True,
                    "fused_bias_fc": False,
                    "pad_vocab_size_multiple": 8,
                    "d_model": 2048,
                    "vocab_size": 50277,
                    "rms_norm": True 
                })


            models["based"] =  ModelConfig(
                    _target_="based.models.gpt.GPTLMHeadModel",
                    config={
                        "_target_": "based.models.gpt.GPT2MixerConfig",
                        "alt_mixer_layers": [1, 6, 11, 16, 21, 27, 33],
                        "alt_mixer_2_layers": [2, 7, 12, 17, 22, 28, 34],

                        "alt_mixer_2": {
                            "_target_": "based.models.mixers.slide_attention.SlidingAttention",
                            "window_size": 128, 
                            "num_heads": 16,
                            "causal": True,
                        },

                        "mixer": {
                            "_target_": "based.models.mixers.convolution.BaseConv",
                            "l_max": PREFILL_SIZE,
                            "kernel_sizes": [3],
                            "expand_proj": 4,
                            "use_cuda": True,
                        },

                        "alt_mixer": {
                            "_target_": "based.models.mixers.linear_attention.LinearAttention",
                            "l_max": PREFILL_SIZE,
                            "feature_dim": 16,
                            "feature_map": {
                                "_target_": "based.models.mixers.linear_attention.TaylorExp",
                                "input_dim": 16,
                            },
                            "head_dim": 64,
                            "num_heads": 28,
                            "parallel_implementation": "tk",
                            "batch_size": batch_size,
                        }, 

                        "n_embd": 1792,
                        "n_head": 16,
                        "n_layer": 36,
                        "rms_norm": True,
                        "special_initializer": True,
                        "fused_mlp": False,
                        "attn_pdrop": 0,
                        "embd_pdrop": 0,
                        "n_positions": PREFILL_SIZE,
                        "resid_pdrop": 0,
                        "mlp_fc1_bias": False,
                        "mlp_fc2_bias": False,
                        "fused_bias_fc": True,
                        "out_proj_bias": False,
                        "qkv_proj_bias": False,
                        "use_flash_attn": True,
                        "residual_in_fp32": True,
                        "activation_function": "swiglu",    # flagging,
                        "rotary_emb_fraction": 1,        # flagging -- 0.5 for the other model.
                        "fused_dropout_add_ln": False,
                        "max_position_embeddings": 0,   # flagging -- not RoPE,
                        "pad_vocab_size_multiple": 8,
                        "reorder_and_upcast_attn": False,
                        "scale_attn_by_inverse_layer_idx": False,
                        "n_inner": 1792 * 2,             
                })


            models["flash-attention"] = ModelConfig(
                    _target_="based.models.gpt.GPTLMHeadModel",
                    config={
                        "_target_": "based.models.gpt.GPT2Config",
                        "reorder_and_upcast_attn": False,
                        "scale_attn_by_inverse_layer_idx": False,
                        "n_positions": PREFILL_SIZE ,
                        "n_embd": 1680,
                        "n_head": 24,
                        "n_layer": 36,
                        "residual_in_fp32": True,
                        "use_flash_attn": True,
                        "fused_dropout_add_ln": False,
                        "fused_mlp": False,
                        "fused_bias_fc": True,
                        "pad_vocab_size_multiple": 16,
                        "rms_norm": True,
                        "attn_pdrop": 0,
                        "embd_pdrop": 0,
                        "resid_pdrop": 0,
                        "mlp_fc1_bias": False,
                        "mlp_fc2_bias": False,
                        "out_proj_bias": False,
                        "qkv_proj_bias": False,
                        "activation_function": "swiglu",
                        "rotary_emb_fraction": 0.5,
                        "max_position_embeddings": 0,
                        "ff_mult": 4,
                    }
                )


            for name, model in models.items():
                print(name)
                n_generated_tokens=0
                config = dict(
                    config=model,
                    n_generated_tokens=n_generated_tokens,
                    batch_size=batch_size,
                    prefill_size=PREFILL_SIZE,
                    fn="forward",
                    repeats=10,
                    name=name,
                    info={
                        "train_view": model.config.get("alt_mixer", {}).get("train_view", None),
                        "ms": model.config.get("alt_mixer", {}).get("mem_save", None),
                        "kernel_size": model.config.get("mixer", {}).get("kernel_sizes", None),
                        "num_heads": model.config.get("alt_mixer", {}).get("num_heads", None),
                        "feature_dim": model.config.get("alt_mixer", {}).get("feature_dim", None),
                    }
                )
                configs.append(config)
