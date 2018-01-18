import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def save_weights(module, path):
    if isinstance(module, DataParallel) or isinstance(module, DistributedDataParallel):
        torch.save(module.module.state_dict(), path)
    else:
        torch.save(module.state_dict(), path)
