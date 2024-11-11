import torch

def get_x_hpu(x_numpy):
    from habana_frameworks.torch.utils.library_loader import load_habana_module

    load_habana_module()

    x_hpu = torch.from_numpy(x_numpy).to("hpu")
    return x_hpu


def is_hpu_device(device: torch.device):
    return device in (torch.device("hpu:0"), torch.device("hpu"))
