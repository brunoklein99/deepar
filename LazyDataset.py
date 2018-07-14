import torch
import torch.utils.data as data
import numpy as np
from data_load import get_window_x_z_at_i_t


class LazyDataset(data.Dataset):
    def __init__(self, meta):
        self.i = meta['i']
        self.t = meta['t']
        self.v = meta['v']
        self.p = meta['p']
        self.s = meta['s']
        self.meta = meta
        self.wlen = meta['wlen']
        assert len(self.i) == len(self.t)
        assert len(self.i) == len(self.p)

    def __len__(self):
        return len(self.i)

    def __getitem__(self, index):
        i, t = self.i[index], self.t[index]
        x, z = get_window_x_z_at_i_t(self.meta, i, t, self.wlen)
        x, z = np.array(x, dtype=np.float32), np.array(z, dtype=np.float32)
        x = torch.from_numpy(x).float()
        z = torch.from_numpy(z).float()
        v = torch.from_numpy(self.v[i, :, :]).float()
        return x, z, v
