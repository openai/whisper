import torch
from habana_frameworks.torch.utils.library_loader import load_habana_module

load_habana_module()

def get_x_hpu(x_numpy):
    x_hpu = torch.from_numpy(x_numpy).to("hpu")
    return x_hpu
