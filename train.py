import numpy as np
import torch
import settings
import math
from model import Net
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from DefaultDataset import DefaultDataset
from data_load import load_parts


def save_model(filename, model):
    state = {'model': model}
    torch.save(state, filename)


def load_model(filename):
    return torch.load(filename)['model']


# https://www.johndcook.com/blog/2008/04/24/how-to-calculate-binomial-probabilities/
def neg_bin_loss(z, mean, alpha):
    r = 1 / alpha
    ma = mean * alpha
    pdf = torch.lgamma(z + r)
    pdf -= torch.lgamma(z + 1)
    pdf -= torch.lgamma(r)
    pdf += r * torch.log(1 / (1 + ma))
    pdf += z * torch.log(ma / (1 + ma))
    # loss = torch.exp(pdf)
    # loss = torch.log(pdf)
    loss = pdf
    loss = torch.sum(loss)
    loss = - loss

    return loss


if __name__ == '__main__':

    np.random.seed(101)
    torch.manual_seed(101)

    data = load_parts()

    x = data['x']
    z = data['z']
    v = data['v']
    p = data['p']

    dataset = DefaultDataset(x, z, v, p)

    loader = DataLoader(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    _, _, x_dim = x.shape
    model = Net(x_dim)
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch {} batch {}/{} loss: {}'.format(epoch, i, len(loader), loss))

    save_model('models/1', model)
