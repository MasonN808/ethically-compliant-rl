import torch
from torch.distributions.utils import _standard_normal

# CPU
# torch.manual_seed(0)
def seed_test():
    # print(torch.distributions.Uniform(torch.tensor(0., device='cpu'), torch.tensor(1., device='cpu')).rsample((5,)))
    # print(_standard_normal(torch.Size([1, 2]), dtype=torch.float32, device="cpu"))
    # print(torch.distributions.Uniform(torch.tensor(0., device='cpu'), torch.tensor(1., device='cpu')).rsample((5,)))
    # print(_standard_normal(torch.Size([1, 2]), dtype=torch.float32, device="cpu"))
    # torch.rand(1).item()
    # print(torch.distributions.Uniform(torch.tensor(0., device='cpu'), torch.tensor(1., device='cpu')).rsample((5,)))
    print(_standard_normal(torch.Size([1, 2]), dtype=torch.float32, device="cpu"))
    print(_standard_normal(torch.Size([1, 2]), dtype=torch.float32, device="cpu"))


