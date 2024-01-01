from typing import Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class TrainCfg:
    # general task params
    batch_size: int = 512
    num_envs: int = 1

    # Lagrangian Parameters
    constraint_type: list[str] = field(default_factory=lambda: ["speed"])
    cost_threshold: list[float] = field(default_factory=lambda: [8])
    lagrange_multiplier: bool = False
    K_P: float = 1
    K_I: float = 1
    K_D: float = 2