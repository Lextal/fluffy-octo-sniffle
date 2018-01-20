import numpy as np
import numpy.random as rnd
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader


def sort_losses(losses, loader):
    """
        A helper function for rearranging the losses according to their order in the iterator
    """
    if isinstance(losses, torch.FloatTensor):
        losses = losses.numpy()
    result = np.zeros_like(losses)
    divisors = np.zeros_like(losses)
    seq = loader.sampler.__seq__
    for i, index in enumerate(seq):
        result[index] += losses[i]
        divisors[index] += 1
    mean_loss = np.mean(losses[np.where(divisors > 0)])
    result[np.where(divisors == 0)] = mean_loss
    divisors[np.where(divisors == 0)] = 1
    return result / divisors


class HardNegativeSampler(Sampler):
    """
        An implementation of Sample for hard negative mining.
        Performs probabilistic sampling with replacement
        Distribution parameter can be the tensor of per-item losses,
            the order must correspond to the order of items in data_source.
    """

    def __init__(self, data_source, distribution=None):
        super(HardNegativeSampler, self).__init__(data_source)
        self.data_source = data_source
        assert len(self.data_source) != 0
        assert distribution is None or distribution.shape[0] == len(self.data_source)
        self.distribution = np.ones(len(self.data_source)) if distribution is None else distribution
        if isinstance(self.distribution, np.ndarray):
            self.distribution = torch.from_numpy(self.distribution)
        self.distribution /= torch.sum(self.distribution)
        self.__seq__ = rnd.choice(list(range(len(self))), size=len(self), replace=True, p=self.distribution.numpy())

    def __iter__(self):
        return iter(self.__seq__)

    def __len__(self):
        return len(self.data_source)


class SimpleDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
