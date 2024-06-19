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
)
tokenizer = gpt.tokenizer
# too many processes crashes, probably memory issue
tokenized = chunk_and_tokenize(dataset, tokenizer, num_proc=8)


layer_input_hooks = ['blocks.{i}.hook_resid_pre' for i in range(gpt.cfg.n_layers)]

cfg = TrainConfig(
    SaeConfig(gpt.config.hidden_size),
    batch_size=16,
    hooks=layer_input_hooks,
    model_kwargs={"fast_ssm": True, "fast_conv": True}
)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()
