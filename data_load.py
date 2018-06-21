import pandas as pd
import numpy as np


def count_positive(v):
    count = 0
    for i in range(len(v)):
        if v[i] > 0:
            count += 1
    return count


def get_keep_indexes(x):
    indexes = []
    for i in range(len(x)):
        if count_positive(x[i, :]) < 10:
            continue
        if count_positive(x[i, :15]) < 1:
            continue
        if count_positive(x[i, -15:]) < 1:
            continue
        indexes.append(i)
    return indexes


# http://isiarticles.com/bundles/Article/pre/pdf/20710.pdf (reference from DeepAR)
# end of page 1, right below abstract
def load_parts(debug=False):
    df = pd.read_csv('data/carparts.csv')

    # "out of 2509 series with complete records for 51 months"
    df = df.dropna()

    parts = df.values
    parts = parts[:, 1:]

    indexes = get_keep_indexes(parts)
    parts = parts[indexes]
    # parts /= np.max(parts)

    # x and y are the same time series, with y shifted by 1
    # carparts dataset is composed of 51 months, minus 1 for x, y shifting
    x = parts[:, :50]
    y = parts[:, 1:51]

    assert x.shape == y.shape

    # last 8 months are used for validation
    valid_x = x[:, -8:]
    # from these 8, the first 4 are sequence encoding (past) and the last 4 are predicting (future)
    valid_y = y[:, -4:]
    # add feature dimension to decoder output
    valid_y = np.expand_dims(valid_y, axis=-1)

    # as above, first 4 months are the encoder inp
    enc_x_valid = valid_x[:, :4]
    # as above, last 4 months are decoder inp
    dec_x_valid = valid_x[:, -4:]

    # add feature dimension
    enc_x_valid = np.expand_dims(enc_x_valid, axis=-1)
    dec_x_valid = np.expand_dims(dec_x_valid, axis=-1)

    # tmp, hold all training related data
    train_x = []
    train_y = []

    # first 42 months are used for training, a training sample is generated
    # from each 8 month window_x
    x = np.expand_dims(x[:, :42], axis=-1)
    y = np.expand_dims(y[:, :42], axis=-1)

    # for each sequence
    for i in range(len(x)):
        if debug:
            print('parts     : ', list(np.squeeze(parts[i])))
            print('parts     : ', list(np.squeeze(parts[i])))
            print('sequence x: ', list(np.squeeze(x[i])))
            print('sequence y: ', list(np.squeeze(y[i])))
        # for each 8 month sequence
        for j in range(42 - 8 + 1):
            # adds 8 months to list
            if debug:
                print('subseq ', j)
            window_x = x[i, j:j+8]
            window_y = y[i, j+4:j+8]
            if debug:
                print('x: ', list(np.squeeze(window_x)))
                print('y:                     ', list(np.squeeze(window_y)))
            train_x.append(np.expand_dims(window_x, axis=0))
            train_y.append(np.expand_dims(window_y, axis=0))

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    assert len(train_x) == len(train_y)

    # same as above
    enc_x_train = train_x[:, :4]
    dec_x_train = train_x[:, -4:]

    return enc_x_train, dec_x_train, train_y, enc_x_valid, dec_x_valid, valid_y
