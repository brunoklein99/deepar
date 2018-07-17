# import matplotlib.pyplot as plt
from math import ceil
from os import makedirs, listdir
from os.path import isdir, join
from random import seed

import numpy as np
import torch
from sklearn.model_selection import KFold
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


def pred(model, enc_x, enc_z, dec_x, dec_v):
    N, T, _ = dec_x.shape
    Z = torch.zeros(N, T, 1)
    if settings.USE_CUDA:
        Z = Z.cuda()
    nbatches = int(ceil(N / settings.BATCH_SIZE))
    nsamples = 25
    for idx_batch in range(nbatches):
        lo = idx_batch * settings.BATCH_SIZE
        hi = lo + settings.BATCH_SIZE
        for idx_sample in range(nsamples):
            if idx_sample % 10 == 0:
                print('running inference batch {}/{} sample {}'.format(idx_batch, nbatches - 1, idx_sample))
            Z[lo:hi] += model.forward_infer(
                enc_x[lo:hi],
                enc_z[lo:hi],
                dec_x[lo:hi],
                dec_v[lo:hi]
            )
    Z /= nsamples
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

    enc_len = data['enc_len']
    dec_len = data['dec_len']
    x_train_valid = data['x']
    z_train_valid = data['z']
    v_train_valid = data['v']
    # p = data['p']
    # enc_x = data['enc_x']
    # enc_z = data['enc_z']
    # dec_x = data['dec_x']
    # dec_z = data['dec_z']
    dec_v = data['dec_v']
    test_enc_x = data['test_enc_x']
    test_enc_z = data['test_enc_z']
    test_dec_x = data['test_dec_x']

    # dataset = DefaultDataset(x, z, v, p)

    # enc_x = torch.from_numpy(enc_x).float()
    # enc_z = torch.from_numpy(enc_z).float()
    # dec_x = torch.from_numpy(dec_x).float()
    # dec_z = torch.from_numpy(dec_z).float()
    dec_v = torch.from_numpy(dec_v).float()
    test_enc_x = torch.from_numpy(test_enc_x).float()
    test_enc_z = torch.from_numpy(test_enc_z).float()
    test_dec_x = torch.from_numpy(test_dec_x).float()
    if settings.USE_CUDA:
        # enc_x = enc_x.cuda()
        # enc_z = enc_z.cuda()
        # dec_x = dec_x.cuda()
        # dec_z = dec_z.cuda()
        dec_v = dec_v.cuda()
        test_enc_x = test_enc_x.cuda()
        test_enc_z = test_enc_z.cuda()
        test_dec_x = test_dec_x.cuda()

    _, _, x_dim = x_train_valid.shape

    nsplit = 5
    fold = KFold(n_splits=nsplit)
    for nfold, (train, valid) in enumerate(fold.split(x_train_valid)):
        x_train = x_train_valid[train]
        x_valid = torch.from_numpy(x_train_valid[valid]).float()
        if settings.USE_CUDA:
            x_valid = x_valid.cuda()

        z_train = z_train_valid[train]
        z_valid = torch.from_numpy(z_train_valid[valid]).float()
        if settings.USE_CUDA:
            z_valid = z_valid.cuda()

        v_train = v_train_valid[train]
        v_valid = torch.from_numpy(v_train_valid[valid]).float()
        if settings.USE_CUDA:
            v_valid = v_valid.cuda()

        enc_x = x_valid[:, :enc_len, :]
        enc_z = z_valid[:, :enc_len, :]
        dec_x = x_valid[:, enc_len:, 13:]
        dec_z = z_valid[:, enc_len:, :]

        model = NegBinNet(x_dim)
        if settings.USE_CUDA:
            model = model.cuda()

        dataset = DefaultDataset(
            x_train,
            z_train,
            v_train
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=settings.BATCH_SIZE,
            shuffle=True
        )

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

                loss = model.loss(z, m, a)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print('fold {} epoch {} batch {}/{} loss: {}'.format(nfold, epoch, i, len(loader), loss))
            z = pred(model, enc_x, enc_z, dec_x, v_valid)
            metric = smape(z, dec_z)
            print('smape', metric)
            dirname = 'checkpoints/{}'.format(nfold)
            makedirs(dirname, exist_ok=True)
            save_model('{}/checkpoint-{:2f}'.format(dirname, metric), model)

        del model
        del x_valid
        del z_valid

    N, T, _ = test_dec_x.shape
    test_dec_z = torch.zeros(N, T, 1)
    if settings.USE_CUDA:
        test_dec_z = test_dec_z.cuda()
    folders = [x for x in listdir('checkpoints') if isdir(join('checkpoints', x))]
    for fold_dir in folders:
        fold_files = listdir('checkpoints/{}'.format(fold_dir))
        assert len(fold_files) > 0
        fold_files = sorted(fold_files)
        best_file = 'checkpoints/{}/{}'.format(fold_dir, fold_files[0])
        print('best file {}'.format(best_file))
        model = load_model(best_file)
        z = pred(model, test_enc_x, test_enc_z, test_dec_x, dec_v)
        test_dec_z += z
    test_dec_z = test_dec_z.cpu().detach().numpy()
    test_dec_z /= len(folders)
    write_submission('submissions/submission.csv', test_dec_z)
