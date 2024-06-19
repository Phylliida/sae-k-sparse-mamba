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
gpt = HookedMamba.from_pretrained(
    MODEL,
    device='cuda'
)
tokenizer = gpt.tokenizer
# too many processes crashes, probably memory issue
tokenized = chunk_and_tokenize(dataset, tokenizer, num_proc=8)


layers = [0]
layer_input_hooks = [f'blocks.{i}.hook_resid_pre' for i in layers]

cfg = TrainConfig(
    sae=SaeConfig(),
    d_in=gpt.cfg.d_model,
    batch_size=32,
    hooks=layer_input_hooks,
    model_kwargs={"fast_ssm": True, "fast_conv": True, "stop_at_layer": 1}
)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()
