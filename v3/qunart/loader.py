import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class ModelLoader:
    """Load any Hugging Face causal LM and its tokenizer."""

    def __init__(self, device: str = "auto", torch_dtype: str = "float16"):
        self.device = device
        self.torch_dtype = torch_dtype

    def _dtype(self):
        if self.torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if self.torch_dtype == "float16":
            return torch.float16
        return torch.float32

    def load(self, model_path: str):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=self._dtype(),
            device_map=self.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        return model, tokenizer, config
