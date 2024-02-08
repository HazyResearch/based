from transformers import AutoTokenizer

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from based.models.gpt import GPTLMHeadModel


@register_model("based_hf")
class BasedLMWrapper(HFLM):
    def __init__(
            self, 
            checkpoint_name: str, 
            base_checkpoint_dir: str, 
            device: str = "cuda",
            **kwargs
        ) -> None:

        if "backend" in kwargs:
            # based currently only supports causal models
            assert kwargs["backend"] == "causal"

        self.checkpoint_name = checkpoint_name
        self.base_checkpoint_dir = base_checkpoint_dir

        model = GPTLMHeadModel.from_pretrained_hf(pretrained_model_name='hazyresearch/based-1.3b', device=device)

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