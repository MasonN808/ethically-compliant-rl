from typing import Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class TrainCfg:
    # general task params
    epochs: int = 150
    total_timesteps: int = 100000