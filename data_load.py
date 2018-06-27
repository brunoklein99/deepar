import datetime

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from math import pi, sin, cos


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


def sample(time_offset, series, gran='monthly', seq_len=8, debug=False):
    assert len(time_offset) == len(series)
    N, T = series.shape
    X = []
    Z = []
    for i in range(N):
        if debug:
            print('i =', i + 1)
            print('z_{i, 1:T}  : ', list(series[i]))
        for t_window in range(T - seq_len + 1):
            if debug:
                print('t_window: ', t_window)
            x = []
            z = []
            for t_time_step in range(seq_len):

                f = []

                t = t_window + t_time_step

                z_prev = 0.0
                z_curr = series[i, t]
                if t > 0:
                    z_prev = series[i, t - 1]

                f.append(z_prev)

                if debug:
                    print('t_time_step : ', t_time_step)
                    print('t           : ', t)
                    print('z_{i, t - 1}: ', z_prev)
                    print('z_{i, t}    : ', z_curr)

                f.append(t / (T - 1))
                f.append(t_time_step / (seq_len - 1))

                if gran == 'monthly':
                    d = time_offset[i] + relativedelta(months=t)
                    f.append(sin(2 * pi * ((d.month - 1) / 11)))
                    f.append(cos(2 * pi * ((d.month - 1) / 11)))
                else:
                    raise Exception('unknown gran')

                x.append(f)
                z.append([z_curr])

            X.append(x)
            Z.append(z)

    X = np.array(X)
    Z = np.array(Z)
    _, T, _ = Z.shape

    v = 1 + np.mean(Z, axis=1)
    v = np.expand_dims(v, axis=-1)

    Z /= v

    p = np.squeeze(v / np.sum(v))

    return X, Z, v, p


def load_parts(debug=False):
    df = pd.read_csv('data/carparts.csv')

    df = df.dropna()

    p = df.values
    p = p[:, 1:]

    indexes = get_keep_indexes(p)

    p = p[indexes]

    z = p

    z = z[:, 1:51]

    assert len(z) == len(p)

    if debug:
        for i in range(len(z)):
            print('{}th sample'.format(i + 1))
            print('p ', list(p[i]))
            print('z ', list(z[i]))

    z = z[:, :42]

    time_offset = [datetime.date(1998, 1, 1)] * len(z)

    return sample(time_offset, z, seq_len=8, debug=debug)
