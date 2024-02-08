from dataclasses import dataclass

@dataclass
class DistillationConfig:

    model_name: str = "microsoft/phi-1"