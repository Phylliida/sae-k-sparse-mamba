from dataclasses import dataclass
from simple_parsing import list_field, Serializable, field

from . import __version__


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """
    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    k: int = 32
    """Number of nonzero features."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    d_in: int = None
    """Size of input to model"""

    batch_size: int = 8
    """Batch size measured in sequences."""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    lr: float | None = None
    """Base LR. If None, it is automatically chosen based on the number of latents."""

    lr_warmup_steps: int = 1000
 
    auxk_alpha: float = 1 / 32
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    hooks: list[str] = list_field()
    """List of hook names that provide data to train SAEs on."""

    distribute_hooks: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    model_kwargs: dict = field(default_factory = lambda: {})
    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1
