import os
import torch
from torch.distributions.utils import _standard_normal
from seed_2 import seed_test

seed = 7
# os.environ['PYTHONHASHSEED'] = str(seed)
# CPU
# torch.manual_seed(seed)
print(_standard_normal(torch.Size([1, 2]), dtype=torch.float32, device="cpu"))
print(_standard_normal(torch.Size([1, 2]), dtype=torch.float32, device="cpu"))

seed_test()