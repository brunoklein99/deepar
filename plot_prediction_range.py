import torch
import numpy as np
import matplotlib.pyplot as plt

from data_load import load_elem


def load_model(filename):
    return torch.load(filename)['model']


if __name__ == '__main__':
    _, data = load_elem()

    # x = data['x']
    # z = data['z']
    # v = data['v']
    # p = data['p']
    enc_x = data['enc_x']
    enc_z = data['enc_z']
    dec_x = data['dec_x']
    dec_z = data['dec_z']
    dec_v = data['dec_v']

    enc_x = torch.from_numpy(enc_x).float()
    enc_z = torch.from_numpy(enc_z).float()
    dec_x = torch.from_numpy(dec_x).float()
    dec_z = torch.from_numpy(dec_z).float()
    dec_v = torch.from_numpy(dec_v).float()

    model = load_model('models/0-23-13872.05')
    model = model.cpu()

    Z = []

    for _ in range(50):
        z = model.forward_infer(enc_x, enc_z, dec_x, dec_v)
        z = z.detach().numpy()
        z = np.expand_dims(z, axis=0)
        Z.append(z)
    Z = np.concatenate(Z)

    _, enc_len, _ = enc_z.shape
    _, dec_len, _ = dec_z.shape

    enc_z = enc_z.detach().numpy()
    enc_z = np.mean(enc_z, axis=0)[:, 0].tolist()
    dec_z = np.mean(Z, axis=(0, 1))[:, 0].tolist()
    stddev = np.mean(np.std(Z, axis=0), axis=0)[:, 0].tolist()

    x = range(enc_len + dec_len)
    y = enc_z + dec_z
    y1 = [x if i <= enc_len else x + stddev[i - enc_len] for i, x in enumerate(y)]
    y2 = [x if i <= enc_len else x - stddev[i - enc_len] for i, x in enumerate(y)]

    plt.plot(x, y)
    plt.axvline(x=enc_len, c='red')
    plt.fill_between(x, y1, y2, facecolor='#A5D1EF')
    plt.show()

    print()