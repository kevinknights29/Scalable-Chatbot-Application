from __future__ import annotations

import os

import torch
from dotenv import find_dotenv
from dotenv import load_dotenv
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig


load_dotenv(find_dotenv())


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
        )
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
        )

    def _load_model(self):
        if torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=True,
                device_map="cuda:0",  # or "auto"
                use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            )
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="cpu",
            torch_dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,
            ),
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"),
        )
