import torch
import torch.utils.data as data


class DefaultDataset(data.Dataset):
    def __init__(self, x, z):
        assert len(x) == len(z)
        self.x = x
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i, :, :]).float()
        z = torch.from_numpy(self.z[i, :, :]).float()
        return x, z
