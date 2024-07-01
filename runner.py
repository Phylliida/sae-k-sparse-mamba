import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mamba_lens import HookedMamba
from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize

import random
import numpy as np
import signal
import wandb
import os
import pickle



if not 'JOB_NAME' in os.environ or os.environ['JOB_NAME'] is None or os.environ['JOB_NAME'] == "":
    raise ValueError(f"Need to set environment variable JOB_NAME")


if os.environ['JOB_COMPLETION_INDEX'] is None or os.environ['JOB_COMPLETION_INDEX'] == "":
    index = 0
    print("no index, running as a standalone script?")
else:
    index = int(os.environ['JOB_COMPLETION_INDEX'])

job_name = os.environ['JOB_NAME'] + str(index)
job_name_text = os.environ['JOB_NAME'] + str(index) + ".txt"

if os.path.exists(job_name_text):
    with open(job_name_text, "r") as f:
        wandb_id = f.read().strip()
    print(f"resuming from {wandb_id}")
    resume = True
else:
    wandb_id = wandb.util.generate_id()
    with open(job_name_text, "w") as f:
        f.write(wandb_id)
    print(f"starting new run {wandb_id}")
    resume = False





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


offset = 1
layers = [offset+index]
layer_input_hooks = [f'blocks.{i}.hook_resid_pre' for i in layers]
#layer_input_hooks = [f'hook_embed']


base_lr = 0.0001414213562373095
#try_lrs = [0.000005, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006]
#try_lrs = [0.000005, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006]
#try_lrs = [float(f) for f in ['0.00011', '0.00012', '0.00013', '0.00015', '0.00016', '0.00017', '0.00018']]
#lr = try_lrs[index]
lr = 0.000125 # did some hparam sweep and this seems pretty good
lr = base_lr

cfg = TrainConfig(
    sae=SaeConfig(k=gpt.cfg.d_model//2, # recommended k size
    ),
    d_in=gpt.cfg.d_model,
    batch_size=64,
    hooks=layer_input_hooks,
    model_kwargs={"fast_ssm": True, "fast_conv": True, 'stop_at_layer': max(layers)+1},
    run_name=str(lr) + " " + job_name_text + " ".join(layer_input_hooks),
    grad_acc_steps=8,
    micro_acc_steps=2,
)

class RNGState(object):
    def __init__(self):
        self.torch_state = torch.get_rng_state()
        self.torch_cuda_state = torch.cuda.get_rng_state_all() 
        self.numpy_state = np.random.get_state()
        self.random_state = random.getstate()
    
    def set(self):
        torch.random.set_rng_state(self.torch_state)
        torch.cuda.set_rng_state_all(self.torch_cuda_state)
        np.random.set_state(self.numpy_state)
        random.setstate(self.random_state)

save_path = job_name + ".pkl"

if resume and not os.path.exists(save_path):
    print("no checkpoint found, starting from scratch")
    resume = False

from transformers import get_linear_schedule_with_warmup, PreTrainedModel
import traceback
if resume:
    try:
        with open(save_path, "rb") as f:
            trainer = pickle.load(f)
            trainer.model = gpt # we don't save the model cause no need to
            trainer.initial_rng_state.set()
            # reattach optimizer and scheduler
            optimizer, lr_scheduler = trainer.get_optimizer_and_scheduler(trainer.saes)
            optimizer.load_state_dict(trainer.optimizer.state_dict())
            lr_scheduler.load_state_dict(trainer.lr_scheduler.state_dict())
            del trainer.optimizer
            del trainer.lr_scheduler
            trainer.optimizer, trainer.lr_scheduler = optimizer, lr_scheduler
    except:
        print(traceback.format_exc())
        print(f"failed to load {save_path}, starting from scratch")
        resume = False
if not resume:
    trainer = SaeTrainer(cfg, tokenized, gpt)
    trainer.initial_rng_state = RNGState()
    trainer.initial_rng_state.set()

class InterruptedException(Exception):
    pass

def interrupt_callback(sig_num, stack_frame):
    raise InterruptedException() 

try:
    signal.signal(signal.SIGINT, interrupt_callback)
    signal.signal(signal.SIGTERM, interrupt_callback)
    trainer.fit(wandb_id=wandb_id, resume=resume)
except (KeyboardInterrupt, InterruptedException):
    import pickle
    print("interrupted, saving progress")
    trainer.model = None
    trainer.resume_rng_state = RNGState()
    with open(save_path, "wb") as f:
        pickle.dump(trainer, f)
    raise