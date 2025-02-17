from dataclasses import field, dataclass
from omegaconf import DictConfig
import torch

@dataclass
class ModelConfig:
    _target_: str
    config: DictConfig
    generate_kwargs: dict = field(default_factory=dict)
    dtype: type = torch.bfloat16
