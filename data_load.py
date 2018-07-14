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


def get_x_z_at_i_t(meta, i: int, t: int):
    s = meta['s']
    v = meta['v']
    x = []
    _, T = s.shape
    x.append(s[i, t - 1] / v[i])
    d = meta['datetime_offset']
    g = meta['g']
    if g == 'm':
        d += relativedelta(months=t)
        x.append(sin(2 * pi * ((d.month - 1) / 11)))
        x.append(cos(2 * pi * ((d.month - 1) / 11)))
    elif g == 'h':
        d += relativedelta(hours=t)
        x.append(sin(2 * pi * (d.hour / 23)))
        x.append(cos(2 * pi * (d.hour / 23)))
        weekday = d.weekday()
        x.append(sin(2 * pi * (weekday / 6)))
        x.append(cos(2 * pi * (weekday / 6)))
    elif g == 'd':
        d += relativedelta(days=t)
        x.append(sin(2 * pi * ((d.day - 1) / 30)))
        x.append(cos(2 * pi * ((d.day - 1) / 30)))
        weekday = d.weekday()
        x.append(sin(2 * pi * (weekday / 6)))
        x.append(cos(2 * pi * (weekday / 6)))
        for idx in range(10):
            if idx == meta['shops'][i]:
                x.append(1.0)
            else:
                x.append(0.0)
        for idx in range(50):
            if idx == meta['items'][i]:
                x.append(1.0)
            else:
                x.append(0.0)
    else:
        raise Exception('gran not supported')
    z = s[i, t]
    return x, z


def get_window_x_z_at_i_t(meta, i: int, t_window: int, window_len: int):
    X = []
    Z = []
    for t in range(t_window, t_window + window_len):
        x, z = get_x_z_at_i_t(meta, i, t)
        X.append(x)
        Z.append([z])
    return X, Z


def get_x_z(meta, t_offset: int, length: int, window_length: int):
    s = meta['s']
    v = meta['v']
    assert len(s) == len(v)
    X = []
    Z = []
    V = []
    N, _ = s.shape

    t_end = t_offset + length - window_length + 1
    for i in range(N):
        for t in range(t_offset, t_end):
            x, z = get_window_x_z_at_i_t(meta, i, t, window_length)
            X.append(x)
            Z.append(z)
            V.append([v[i]])
            if t % 500 == 0:
                print('i {}/{} t {}/{}'.format(i, N, t, t_end))

    X = np.array(X)
    Z = np.array(Z)
    V = np.array(V)

    return X, Z, V


def get_i_t(meta, t_offset: int, length: int, window_length: int):
    s = meta['s']
    v = meta['v']
    assert len(s) == len(v)
    I = []
    T = []
    V = []
    N, _ = s.shape

    t_end = t_offset + length - window_length + 1
    for i in range(N):
        for t in range(t_offset, t_end):
            I.append(i)
            T.append(t)
            V.append([v[i]])

    V = np.array(V)

    return I, T, V


def get_x_z_subsample(meta, t_offset: int, length: int, window_length: int, count: int):
    s = meta['s']
    v = meta['v']
    assert len(s) == len(v)
    X = []
    Z = []
    V = []
    N, _ = s.shape

    for c in range(count):
        i = randint(0, len(s) - 1)
        t = randint(t_offset, t_offset + length - window_length + 1)
        x, z = get_window_x_z_at_i_t(meta, i, t, window_length)
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


def get_kaggle_series():
    df = pd.read_csv('data/kaggle_train.csv')
    series = []
    items = []
    shops = []
    for i in range(max(df['item'])):
        for s in range(max(df['store'])):
            serie = df.loc[(df['item'] == i + 1) & (df['store'] == s + 1)]
            serie = list(serie['sales'])
            assert len(serie) == 1826
            series.append(serie)
            items.append(i)
            shops.append(s)
    series = np.array(series)
    datetime_offset = datetime.datetime(2013, 1, 1)
    meta = {
        's': series,
        'items': items,
        'shops': shops,
        'g': 'd',
        'datetime_offset': datetime_offset
    }
    return datetime_offset, meta


def load_kaggle():
    datetime_offset, meta = get_kaggle_series()

    s = meta['s']

    N, T = s.shape

    enc_len = 180
    dec_len = 90
    train_len = T - dec_len - 1

    # first t of the series
    t1 = 1

    # first t of prediction range
    t0 = t1 + train_len

    # first t of encoder (validation)
    t_enc = t0 - enc_len

    v = 1 + np.mean(s[:, t1:t0], axis=1)

    meta['v'] = v

    wlen = enc_len + dec_len
    meta['wlen'] = wlen

    i, t, v_train = get_i_t(
        meta,
        t_offset=t1,
        length=train_len,
        window_length=wlen
    )

    p = np.squeeze(v_train / np.sum(v_train))

    meta['i'] = i
    meta['t'] = t
    meta['p'] = p

    enc_x, enc_z, _ = get_x_z(
        meta,
        t_offset=t_enc,
        length=enc_len,
        window_length=enc_len
    )

    dec_x, dec_z, _ = get_x_z(
        meta,
        t_offset=t0,
        length=dec_len,
        window_length=dec_len
    )

    test_enc_x, test_enc_z, _ = get_x_z(
        meta,
        t_offset=T - enc_len,
        length=enc_len,
        window_length=enc_len
    )

    # mock series, just to build input features of the prediction range
    rnd = np.random.randn(N, T + dec_len)

    meta_mock = meta.copy()
    meta_mock['s'] = rnd

    test_dec_x, _, _ = get_x_z(
        meta_mock,
        t_offset=T,
        length=dec_len,
        window_length=dec_len
    )

    v = np.expand_dims(v, axis=-1)
    v = np.expand_dims(v, axis=-1)

    meta['v'] = v

    data = {
        'meta': meta,
        'enc_x': enc_x,
        'enc_z': enc_z,
        'dec_x': dec_x[:, :, 1:],
        'dec_z': dec_z,
        'dec_v': v,
        'test_enc_x': test_enc_x,
        'test_enc_z': test_enc_z,
        'test_dec_x': test_dec_x[:, :, 1:]
    }

    return datetime_offset, data


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
        t_offset=t1,
        length=train_len,
        window_length=enc_len,
        count=500_000
    )

    p = np.squeeze(v_train / np.sum(v_train))
    v_train = np.expand_dims(v_train, axis=-1)

    enc_x, enc_z, _ = get_x_z(
        s,
        v,
        t_offset=t_enc,
        length=enc_len,
        window_length=enc_len
    )

    dec_x, dec_z, _ = get_x_z(
        s,
        v,
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
        t_offset=t1,
        length=train_len,
        window_length=8,
    )

    p = np.squeeze(v_train / np.sum(v_train))
    v_train = np.expand_dims(v_train, axis=-1)

    enc_x, enc_z, _ = get_x_z(
        s,
        v,
        t_offset=t_enc,
        length=enc_len,
        window_length=enc_len,
    )

    dec_x, dec_z, _ = get_x_z(
        s,
        v,
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
