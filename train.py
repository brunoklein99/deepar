import numpy as np
import torch
from torch.distributions import Poisson, Gamma

import settings
import math
from model import Net
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from DefaultDataset import DefaultDataset
from data_load import load_parts


def rmse(z, mean, alpha):
    r = 1 / alpha
    ma = mean * alpha

    p = ma / (1 + ma)
    beta = (1 - p) / p

    g = Gamma(concentration=r, rate=beta)
    g = g.sample()

    p = Poisson(g)

    z_pred = p.sample()

    rmse = z_pred - z
    rmse = torch.pow(rmse, 2)
    rmse = torch.mean(rmse)
    rmse = torch.sqrt(rmse)

    return rmse


# https://www.johndcook.com/blog/2008/04/24/how-to-calculate-binomial-probabilities/
def neg_bin_loss(z, mean, alpha):
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


if __name__ == '__main__':

    np.random.seed(101)
    torch.manual_seed(101)

    x, z, v, p = load_parts()

    dataset = DefaultDataset(x, z, v, p)

    loader = DataLoader(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    model = Net()
    if settings.USE_CUDA:
        model = model.cuda()

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    for epoch in range(settings.EPOCHS):
        for i, (x, z, v) in enumerate(loader):
            x = Variable(x)
            z = Variable(z)
            v = Variable(v)

            if settings.USE_CUDA:
                x = x.cuda()
                z = z.cuda()
                v = v.cuda()

            m, a = model(x, v)

            loss = neg_bin_loss(z, m, a)
            print('rmse', rmse(
                z.cpu(),
                m.cpu(),
                a.cpu()
            ))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch {} batch {}/{} loss: {}'.format(epoch, i, len(loader), loss))
