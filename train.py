# import matplotlib.pyplot as plt
from random import seed

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import settings
from DefaultDataset import DefaultDataset
from data_load import load_kaggle
from model import GaussianNet, NegBinNet


def save_model(filename, model):
    state = {'model': model}
    torch.save(state, filename)


def load_model(filename):
    return torch.load(filename)['model']


def rmse(z_true, z_pred):
    return float(torch.sqrt(torch.mean(torch.pow(z_pred - z_true, 2))))


def pred(enc_x, enc_z, dec_x, dec_v):
    Z = []
    for i in range(25):
        z = model.forward_infer(enc_x, enc_z, dec_x, dec_v)
        z = z.unsqueeze(0)
        Z.append(z)
    Z = torch.cat(Z)
    Z = torch.mean(Z, dim=0, keepdim=False)
    return Z


def rmse_mean(enc_x, enc_z, dec_x, dec_v):
    Z = pred(enc_x, enc_z, dec_x, dec_v)
    return rmse(dec_z, Z)


def smape(f, a):
    return torch.mean(2 * (torch.abs(a - f) / (torch.abs(a) + torch.abs(f))))


# def plot(results):
#     plt.plot(range(len(results)), results)
#     plt.show()
def write_submission(filename, z):
    N, T, _ = z.shape
    with open(filename, 'w') as f:
        f.write('id,sales\n')
        idx = 0
        for i in range(N):
            for t in range(T):
                f.write('{},{}\n'.format(idx, float(z[i, t])))
                idx += 1


if __name__ == '__main__':

    np.random.seed(101)
    torch.manual_seed(101)
    seed(101)

    _, data = load_kaggle()

    x = data['x']
    z = data['z']
    v = data['v']
    p = data['p']
    enc_x = data['enc_x']
    enc_z = data['enc_z']
    dec_x = data['dec_x']
    dec_z = data['dec_z']
    dec_v = data['dec_v']
    test_enc_x = data['test_enc_x']
    test_enc_z = data['test_enc_z']
    test_dec_x = data['test_dec_x']

    dataset = DefaultDataset(x, z, v, p)

    enc_x = torch.from_numpy(enc_x).float()
    enc_z = torch.from_numpy(enc_z).float()
    dec_x = torch.from_numpy(dec_x).float()
    dec_z = torch.from_numpy(dec_z).float()
    dec_v = torch.from_numpy(dec_v).float()
    test_enc_x = torch.from_numpy(test_enc_x).float()
    test_enc_z = torch.from_numpy(test_enc_z).float()
    test_dec_x = torch.from_numpy(test_dec_x).float()
    if settings.USE_CUDA:
        enc_x = enc_x.cuda()
        enc_z = enc_z.cuda()
        dec_x = dec_x.cuda()
        dec_z = dec_z.cuda()
        dec_v = dec_v.cuda()
        test_enc_x = test_enc_x.cuda()
        test_enc_z = test_enc_z.cuda()
        test_dec_x = test_dec_x.cuda()

    loader = DataLoader(
        dataset=dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    _, _, x_dim = x.shape
    model = NegBinNet(x_dim)
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

            m, a = model(x, v)

            loss = model.loss(z, m, a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('epoch {} batch {}/{} loss: {}'.format(epoch, i, len(loader), loss))

        z = pred(enc_x, enc_z, dec_x, dec_v)
        metric = smape(z, dec_z)
        print('smape', metric)
        test_dec_z = pred(
            test_enc_x,
            test_enc_z,
            test_dec_x,
            dec_v
        )
        test_dec_z = test_dec_z.cpu().detach().numpy()
        write_submission('submissions/submission-{:2f}.csv'.format(metric), test_dec_z)

    save_model('models/1', model)

    # plot(results)

    print('rmse final', rmse_mean(enc_x, enc_z, dec_x, dec_v))
