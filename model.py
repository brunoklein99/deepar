from abc import abstractmethod
from math import pi

from torch.distributions import Gamma, Poisson

import settings
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @abstractmethod
    def forward_ma(self, o, v):
        pass

    @abstractmethod
    def sample(self, m, a):
        pass

    @abstractmethod
    def loss(self, z, m, a):
        pass

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
            z = self.sample(m, a)
            Z.append(z)
            z = z / v
        Z = torch.cat(Z, 1)
        return Z


class NegBinNet(Net):

    # https://www.johndcook.com/blog/2008/04/24/how-to-calculate-binomial-probabilities/
    def loss(self, z, mean, alpha):
        r = 1 / alpha
        ma = mean * alpha
        pdf = torch.lgamma(z + r)
        pdf -= torch.lgamma(z + 1)
        pdf -= torch.lgamma(r)
        pdf += r * torch.log(1 / (1 + ma))
        pdf += z * torch.log(ma / (1 + ma))
        pdf = torch.exp(pdf)

        loss = torch.log(pdf)
        loss = torch.sum(loss)
        loss = - loss

        return loss

    def sample(self, m, a):
        r = 1 / a
        p = (m * a) / (1 + (m * a))
        b = (1 - p) / p
        g = Gamma(r, b)
        g = g.sample()
        p = Poisson(g)
        z = p.sample()
        return z

    def forward_ma(self, o, v):
        m = F.softplus(self.linear_m(o))
        a = F.softplus(self.linear_a(o))
        m = torch.mul(m, v)
        a = torch.div(a, torch.sqrt(v))
        return m, a


class GaussianNet(Net):

    def loss(self, z, m, a):
        v = a * a
        t1 = 2 * pi * v
        t1 = torch.pow(t1, -1 / 2)
        t1 = torch.log(t1)

        t2 = z - m
        t2 = torch.pow(t2, 2)
        t2 = - t2
        t2 = t2 / (2 * v)

        loss = t1 + t2
        # loss = torch.exp(loss)
        # loss = torch.log(loss)
        loss = torch.sum(loss)
        loss = -loss

        return loss

    def sample(self, m, a):
        return torch.normal(m, a)

    def forward_ma(self, o, v):
        m = self.linear_m(o)
        a = F.softplus(self.linear_a(o))
        m = torch.mul(m, v)
        a = torch.mul(a, v)
        return m, a
