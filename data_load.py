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


def get_x_z_at_i_t(s, v, datetime_offset: datetime.datetime, i: int, t: int):
    x = []
    _, T = s.shape
    x.append(s[i, t - 1] / v[i])
    x.append(t / T)
    d = datetime_offset + relativedelta(months=t)
    x.append(sin(2 * pi * ((d.month - 1) / 11)))
    x.append(cos(2 * pi * ((d.month - 1) / 11)))
    z = s[i, t]
    return x, z


def get_window_x_z_at_i_t(s, v, datetime_offset: datetime.datetime, i: int, t_window: int, window_len: int):
    X = []
    Z = []
    for t in range(t_window, t_window + window_len):
        x, z = get_x_z_at_i_t(s, v, datetime_offset, i, t)
        X.append(x)
        Z.append([z])
    return X, Z


def get_x_z(s, v, datetime_offset: datetime.datetime, t_offset: int, length: int, window_length: int):
    X = []
    Z = []
    V = []
    N, _ = s.shape

    for i in range(N):
        for t in range(t_offset, t_offset + length - window_length + 1):
            x, z = get_window_x_z_at_i_t(s, v, datetime_offset, i, t, window_length)
            X.append(x)
            Z.append(z)
            V.append([v[i]])

    X = np.array(X)
    Z = np.array(Z)
    V = np.array(V)

    return X, Z, V


def load_parts():
    df = pd.read_csv('data/carparts.csv')

    df = df.dropna()

    s = df.values

    # no item id
    s = s[:, 1:]

    indexes = get_keep_indexes(s)

    s = s[indexes]

    datetime_offset = datetime.datetime(1998, 1, 1)

    enc_len = 8
    dec_len = 8
    train_len = 42

    # first t of the series
    t1 = 1

    # first t of prediction range
    t0 = t1 + train_len

    # first t of encoder (validation)
    t_enc = t0 - enc_len

    v = 1 + np.mean(s[:, t1:t0], axis=1)

    x, z, v = get_x_z(
        s,
        v,
        datetime_offset,
        t_offset=t1,
        length=train_len,
        window_length=8,
    )

    p = v / np.sum(v)

    enc_x, enc_z, _ = get_x_z(
        s,
        v,
        datetime_offset,
        t_offset=t_enc,
        length=enc_len,
        window_length=enc_len,
    )

    dec_x, dec_z, _ = get_x_z(
        s,
        v,
        datetime_offset,
        t_offset=t0,
        length=dec_len,
        window_length=dec_len
    )

    data = {
        'x': x,
        'z': z,
        'v': v,
        'p': p,
        'enc_x': enc_x,
        'enc_z': enc_z,
        'dec_x': dec_x,
        'dec_z': dec_z,
    }

    return data
