from typing import Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class TrainCfg:
    # general task params
    task: str = "SafetyCarCircle-v0"
    device: str = "cpu"
    # Lagrangian Parameters
    constraint_type: list[str] = field(default_factory=lambda: ["speed"])
    cost_threshold: list[float] = field(default_factory=lambda: [2])
    # K_P: float = 1
    # K_I: float = 1
    # K_D: float = 2
    K_P: float = 0.05
    K_I: float = 0.0005
    K_D: float = 0.1
    # logger params
    logdir: str = "logs"
    project: str = "fast-safe-rl"
    group: Optional[str] = None
    name: Optional[str] = None
    prefix: Optional[str] = "ppol"
    suffix: Optional[str] = ""