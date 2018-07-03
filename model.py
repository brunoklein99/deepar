from torch.distributions import Gamma, Poisson

import settings
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample(m, a):
    r = 1 / a
    p = (m * a) / (1 + (m * a))
    b = (1 - p) / p
    g = Gamma(r, b)
    g = g.sample()
    p = Poisson(g)
    z = p.sample()
    return z


class Net(nn.Module):

    def __init__(self, x_dim):
        super().__init__()
        self.cell = nn.LSTM(
            input_size=x_dim,
            hidden_size=settings.HIDDEN_DIM,
            num_layers=3,
            batch_first=True
        )
        self.linear_m = nn.Linear(
            in_features=settings.HIDDEN_DIM,
            out_features=1
        )
        self.linear_a = nn.Linear(
            in_features=settings.HIDDEN_DIM,
            out_features=1
        )

    def forward_ma(self, o, v):
        m = F.softplus(self.linear_m(o))
        a = F.softplus(self.linear_a(o))
        m = torch.mul(m, v)
        a = torch.div(a, torch.sqrt(v))
        return m, a

    def forward(self, x, v):
        o, (_, _) = self.cell(x)
        m, a = self.forward_ma(o, v)
        return m, a

    def forward_infer(self, enc_x, enc_z, dec_x, v):
        _, (h, c) = self.cell(enc_x)
        _, T, _ = dec_x.shape
        z = enc_z[:, -1, 0]
        z = z.unsqueeze(-1)
        z = z.unsqueeze(-1)
        z = z / v
        Z = []
        for t in range(T):
            x = dec_x[:, t, :]
            x = x.unsqueeze(1)
            x = torch.cat((z, x), 2)
            o, (h, c) = self.cell(x, (h, c))
            m, a = self.forward_ma(o, v)
            z = _sample(m, a)
            Z.append(z)
            z = z / v
        Z = torch.cat(Z, 1)
        return Z
