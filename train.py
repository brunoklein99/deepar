import numpy as np
import torch
import settings
from model import Net
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from DefaultDataset import DefaultDataset
from data_load import load_parts


def gamma(x):
    return torch.exp(torch.lgamma(x))


def neg_bin_pdf(r, k, m):
    t1_n = gamma(r + k)
    t1_d = gamma(k + 1) * gamma(r)
    t1 = t1_n / t1_d

    t2_n = m
    t2_d = r + m
    t2 = t2_n / t2_d
    t2 = torch.pow(t2, k)

    t3_n = r
    t3_d = r + m
    t3 = t3_n / t3_d
    t3 = torch.pow(t3, r)

    pdf = t1 * t2 * t3

    return pdf


def neg_bin_loss(z, m, a):

    inva = 1 / a
    am = a * m

    t1_n = gamma(z + inva)
    t1_d = gamma(z + 1) * gamma(inva)
    t1 = t1_n / t1_d

    t2_n = 1
    t2_d = gamma(1 + am)
    t2 = t2_n / t2_d
    t2 = torch.pow(t2, inva)

    t3_n = am
    t3_d = 1 + am
    t3 = t3_n / t3_d
    t3 = torch.pow(t3, z)

    # pdf = neg_bin_pdf(
    #     r=1/a,
    #     k=z,
    #     m=m,
    # )
    pdf = t1 * t2 * t3
    pdf[pdf > 100] = 100

    loss = torch.log(pdf)
    loss = torch.sum(loss)
    loss = - loss

    return loss


if __name__ == '__main__':
    torch.manual_seed(101)

    x, z = load_parts()

    dataset = DefaultDataset(x, z)

    loader = DataLoader(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    model = Net().cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    for epoch in range(settings.EPOCHS):
        for i, (x, z) in enumerate(loader):
            x = Variable(x).cuda()
            z = Variable(z).cuda()

            m, a = model(x)
            loss = neg_bin_loss(z, m, a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch {} batch {}/{} loss: {}'.format(epoch, i, len(loader), loss))
