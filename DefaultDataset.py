import torch
import torch.utils.data as data
import numpy as np


class DefaultDataset(data.Dataset):
    def __init__(self, x, z, v, p):
        assert len(x) == len(z)
        assert len(x) == len(v)
        assert len(x) == len(p)
        self.x = x
        self.z = z
        self.v = v
        self.p = p

    def __len__(self):
        return len(self.x)

    def __getitem__(self, _):
        r = range(len(self.x))
        i = np.random.choice(r, p=self.p)
        x = torch.from_numpy(self.x[i, :, :]).float()
        z = torch.from_numpy(self.z[i, :, :]).float()
        v = torch.from_numpy(self.v[i, :, :]).float()
        return x, z, v
