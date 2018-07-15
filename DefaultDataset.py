import torch
import torch.utils.data as data
import numpy as np


class DefaultDataset(data.Dataset):
    def __init__(self, x, z, v):
        assert len(x) == len(z)
        assert len(x) == len(v)
        self.x = x
        self.z = z
        self.v = v

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i, :, :]).float()
        z = torch.from_numpy(self.z[i, :, :]).float()
        v = torch.from_numpy(self.v[i, :, :]).float()
        return x, z, v
