from transformers import AutoTokenizer

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("based_hf")
class BasedLMWrapper(HFLM):
    def __init__(
            self, 
            model: str = "based",
            checkpoint_name: str='hazyresearch/based-1.3b', 
            device: str = "cuda",
            **kwargs
        ) -> None:

        assert model in ['based', 'mamba', 'transformer'], print("Model must be one of 'based', 'mamba', or 'transformer'")

        if "backend" in kwargs:
            # based currently only supports causal models
            assert kwargs["backend"] == "causal"

        self.checkpoint_name = checkpoint_name

        if model == "based":
            from based.models.gpt import GPTLMHeadModel
            model = GPTLMHeadModel.from_pretrained_hf(pretrained_model_name=self.checkpoint_name, device=device)
        elif model == "mamba": 
            from based.models.mixer_model import MambaLMHeadModel
            model = MambaLMHeadModel.from_pretrained_hf(pretrained_model_name=self.checkpoint_name, device=device)
        elif model == "transformer":
            from flash_attn import GPTLMHeadModel
            model = GPTLMHeadModel.from_pretrained_hf(pretrained_model_name=self.checkpoint_name, device=device)
        else:
            raise ValueError(f"Unsupported model {model}")

        tokenizer_name = kwargs.get("tokenizer", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        model.device = device

        super().__init__(
            pretrained=model,
            # set appropriate defaults for tokenizer, max length, etc
            backend=kwargs.get("backend", "causal"),
            max_length=kwargs.get("max_length", 2048),
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )