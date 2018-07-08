# import matplotlib.pyplot as plt
from random import seed

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import settings
from DefaultDataset import DefaultDataset
from data_load import load_parts, load_elec
from model import NegBinNet, GaussianNet


def save_model(filename, model):
    state = {'model': model}
    torch.save(state, filename)


def load_model(filename):
    return torch.load(filename)['model']


def rmse(z_true, z_pred):
    return float(torch.sqrt(torch.mean(torch.pow(z_pred - z_true, 2))))


# def plot(results):
#     plt.plot(range(len(results)), results)
#     plt.show()


if __name__ == '__main__':

    np.random.seed(101)
    torch.manual_seed(101)
    seed(101)

    _, data = load_elec()

    x = data['x']
    z = data['z']
    v = data['v']
    p = data['p']
    enc_x = data['enc_x']
    enc_z = data['enc_z']
    dec_x = data['dec_x']
    dec_z = data['dec_z']
    dec_v = data['dec_v']

    dataset = DefaultDataset(x, z, v, p)

    enc_x = torch.from_numpy(enc_x).float()
    enc_z = torch.from_numpy(enc_z).float()
    dec_x = torch.from_numpy(dec_x).float()
    dec_z = torch.from_numpy(dec_z).float()
    dec_v = torch.from_numpy(dec_v).float()
    if settings.USE_CUDA:
        enc_x = enc_x.cuda()
        enc_z = enc_z.cuda()
        dec_x = dec_x.cuda()
        dec_z = dec_z.cuda()
        dec_v = dec_v.cuda()

    loader = DataLoader(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    _, _, x_dim = x.shape
    model = GaussianNet(x_dim)
    if settings.USE_CUDA:
        model = model.cuda()

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    results = []
    for epoch in range(settings.EPOCHS):
        for i, (x, z, v) in enumerate(loader):
            x = Variable(x)
            z = Variable(z)
            v = Variable(v)

            if settings.USE_CUDA:
                x = x.cuda()
                z = z.cuda()
                v = v.cuda()

            if i % 10 == 0:
                z_pred = model.forward_infer(enc_x, enc_z, dec_x, dec_v)
                result = rmse(dec_z, z_pred)
                results.append(result)
                print('rmse', result)

            m, a = model(x, v)

            loss = model.loss(z, m, a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch {} batch {}/{} loss: {}'.format(epoch, i, len(loader), loss))

    save_model('models/1', model)

    # plot(results)

    model.eval()

    Z = []
    for i in range(50):
        z = model.forward_infer(enc_x, enc_z, dec_x, dec_v)
        z = z.numpy()
        z = np.expand_dims(z, axis=0)
        Z.append(z)
    Z = np.concatenate(Z)
    Z = np.mean(Z, axis=0)
    print('rmse', rmse(dec_z, Z))
