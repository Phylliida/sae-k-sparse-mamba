import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mamba_lens import HookedMamba
from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize

MODEL = "state-spaces/mamba-370m"
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer)

gpt = HookedMamba.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(
    SaeConfig(gpt.config.hidden_size), batch_size=16
)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()