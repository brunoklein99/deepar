import datetime
from random import randint

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


def get_x_z_at_i_t(s, v, datetime_offset: datetime.datetime, i: int, t: int, gran):
    x = []
    _, T = s.shape
    x.append(s[i, t - 1] / v[i])
    x.append(t / T)
    d = datetime_offset
    if gran == 'm':
        d += relativedelta(months=t)
        x.append(sin(2 * pi * ((d.month - 1) / 11)))
        x.append(cos(2 * pi * ((d.month - 1) / 11)))
    elif gran == 'h':
        d += relativedelta(hours=t)
        x.append(sin(2 * pi * (d.hour / 23)))
        x.append(cos(2 * pi * (d.hour / 23)))
        weekday = d.weekday()
        x.append(sin(2 * pi * (weekday / 6)))
        x.append(cos(2 * pi * (weekday / 6)))
    else:
        raise Exception('gran not supported')
    z = s[i, t]
    return x, z


def get_window_x_z_at_i_t(s, v, datetime_offset: datetime.datetime, i: int, t_window: int, window_len: int, gran):
    X = []
    Z = []
    for t in range(t_window, t_window + window_len):
        x, z = get_x_z_at_i_t(s, v, datetime_offset, i, t, gran)
        X.append(x)
        Z.append([z])
    return X, Z


def get_x_z(s, v, datetime_offset: datetime.datetime, t_offset: int, length: int, window_length: int, gran='m'):
    assert len(s) == len(v)
    X = []
    Z = []
    V = []
    N, _ = s.shape

    for i in range(N):
        for t in range(t_offset, t_offset + length - window_length + 1):
            x, z = get_window_x_z_at_i_t(s, v, datetime_offset, i, t, window_length, gran)
            X.append(x)
            Z.append(z)
            V.append([v[i]])

    X = np.array(X)
    Z = np.array(Z)
    V = np.array(V)

    return X, Z, V


def get_x_z_subsample(s, v, datetime_offset: datetime.datetime, t_offset: int, length: int, window_length: int, count: int, gran='m'):
    assert len(s) == len(v)
    X = []
    Z = []
    V = []
    N, _ = s.shape

    for c in range(count):
        i = randint(0, len(s) - 1)
        t = randint(t_offset, t_offset + length - window_length + 1)
        x, z = get_window_x_z_at_i_t(s, v, datetime_offset, i, t, window_length, gran)
        X.append(x)
        Z.append(z)
        V.append([v[i]])
        if c % 1000 == 0:
            print('sampling {}/{}'.format(c, count))

    X = np.array(X)
    Z = np.array(Z)
    V = np.array(V)

    return X, Z, V


def get_parts_series():
    df = pd.read_csv('data/carparts.csv')

    df = df.dropna()

    s = df.values

    # no item id
    s = s[:, 1:]

    indexes = get_keep_indexes(s)

    s = s[indexes]

    datetime_offset = datetime.datetime(1998, 1, 1)

    return datetime_offset, s


def get_elec_series():
    df = pd.read_csv('data/elec.csv')

    datetime_offset = datetime.datetime(2000, 1, 1)

    s = df.values.T

    return datetime_offset, s


def load_elec():
    datetime_offset, s = get_elec_series()

    _, T = s.shape

    enc_len = 168
    dec_len = 24
    train_len = T - dec_len - 1

    # first t of the series
    t1 = 1

    # first t of prediction range
    t0 = t1 + train_len

    # first t of encoder (validation)
    t_enc = t0 - enc_len

    v = 1 + np.mean(s[:, t1:t0], axis=1)

    gran = 'h'

    x_train, z_train, v_train = get_x_z_subsample(
        s,
        v,
        datetime_offset,
        t_offset=t1,
        length=train_len,
        window_length=enc_len + dec_len,
        count=100_000,
        gran=gran
    )

    p = np.squeeze(v_train / np.sum(v_train))
    v_train = np.expand_dims(v_train, axis=-1)

    enc_x, enc_z, _ = get_x_z(
        s,
        v,
        datetime_offset,
        t_offset=t_enc,
        length=enc_len,
        window_length=enc_len,
        gran=gran
    )

    dec_x, dec_z, _ = get_x_z(
        s,
        v,
        datetime_offset,
        t_offset=t0,
        length=dec_len,
        window_length=dec_len,
        gran=gran
    )

    v = np.expand_dims(v, axis=-1)
    v = np.expand_dims(v, axis=-1)

    data = {
        'x': x_train,
        'z': z_train,
        'v': v_train,
        'p': p,
        'enc_x': enc_x,
        'enc_z': enc_z,
        'dec_x': dec_x[:, :, 1:],
        'dec_z': dec_z,
        'dec_v': v,
    }

    return datetime_offset, data


def load_parts():
    datetime_offset, s = get_parts_series()

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

    x_train, z_train, v_train = get_x_z(
        s,
        v,
        datetime_offset,
        t_offset=t1,
        length=train_len,
        window_length=8,
    )

    p = np.squeeze(v_train / np.sum(v_train))
    v_train = np.expand_dims(v_train, axis=-1)

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

    v = np.expand_dims(v, axis=-1)
    v = np.expand_dims(v, axis=-1)

    data = {
        'x': x_train,
        'z': z_train,
        'v': v_train,
        'p': p,
        'enc_x': enc_x,
        'enc_z': enc_z,
        'dec_x': dec_x[:, :, 1:],
        'dec_z': dec_z,
        'dec_v': v,
    }

    return datetime_offset, data
