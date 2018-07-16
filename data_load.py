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


def get_x_z_at_i_t(meta, i: int, t: int, out_i: int, out_t: int, out_x, out_z=None, use_lag_feat=True):
    
    def set_get_index(idx, vector, value):
        vector[out_i, out_t, idx] = value
        return idx + 1
    
    s = meta['s']
    v = meta['v']
    assert len(s) == len(v)
    _, T = s.shape
    index_x = 0

    if use_lag_feat:
        t_lag = t - 7
        if t_lag >= 0:
            index_x = set_get_index(index_x, out_x, np.min(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.max(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.median(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.mean(s[i, t_lag:t]) / v[i])
        else:
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)

        t_lag = t - 30
        if t_lag >= 0:
            index_x = set_get_index(index_x, out_x, np.min(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.max(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.median(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.mean(s[i, t_lag:t]) / v[i])
        else:
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)

        t_lag = t - 60
        if t_lag >= 0:
            index_x = set_get_index(index_x, out_x, np.min(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.max(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.median(s[i, t_lag:t]) / v[i])
            index_x = set_get_index(index_x, out_x, np.mean(s[i, t_lag:t]) / v[i])
        else:
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)
            index_x = set_get_index(index_x, out_x, 0.0)

        index_x = set_get_index(index_x, out_x, s[i, t - 1] / v[i])
    index_x = set_get_index(index_x, out_x, t / (T + 89))

    d = meta['datetime_offset']
    g = meta['g']
    if g == 'm':
        d += relativedelta(months=t)
        index_x = set_get_index(index_x, out_x, sin(2 * pi * ((d.month - 1) / 11)))
        index_x = set_get_index(index_x, out_x, cos(2 * pi * ((d.month - 1) / 11)))
    elif g == 'h':
        d += relativedelta(hours=t)
        index_x = set_get_index(index_x, out_x, sin(2 * pi * (d.hour / 23)))
        index_x = set_get_index(index_x, out_x, cos(2 * pi * (d.hour / 23)))
        weekday = d.weekday()
        index_x = set_get_index(index_x, out_x, sin(2 * pi * (weekday / 6)))
        index_x = set_get_index(index_x, out_x, cos(2 * pi * (weekday / 6)))
    elif g == 'd':
        d += relativedelta(days=t)
        index_x = set_get_index(index_x, out_x, sin(2 * pi * ((d.day - 1) / 30)))
        index_x = set_get_index(index_x, out_x, cos(2 * pi * ((d.day - 1) / 30)))
        weekday = d.weekday()

        index_x = set_get_index(index_x, out_x, sin(2 * pi * (weekday / 6)))
        index_x = set_get_index(index_x, out_x, cos(2 * pi * (weekday / 6)))

        index_x = set_get_index(index_x, out_x, (d.year - 2013) / 5)
        index_x = set_get_index(index_x, out_x, sin(2 * pi * ((d.month - 1) / 11)))
        index_x = set_get_index(index_x, out_x, cos(2 * pi * ((d.month - 1) / 11)))

        for idx in range(10):
            if idx == meta['shops'][i]:
                index_x = set_get_index(index_x, out_x, 1.0)
            else:
                index_x = set_get_index(index_x, out_x, 0.0)
        for idx in range(50):
            if idx == meta['items'][i]:
                index_x = set_get_index(index_x, out_x, 1.0)
            else:
                index_x = set_get_index(index_x, out_x, 0.0)
    else:
        raise Exception('gran not supported')
    _, _, x_dim = out_x.shape
    assert index_x == x_dim
    if out_z is not None:
        out_z[out_i, out_t, 0] = s[i, t]


def get_window_x_z_at_i_t(meta, i: int, t_window: int, window_len: int, out_i: int, out_x, out_z=None, use_lag_feat=True):
    for t in range(t_window, t_window + window_len):
        get_x_z_at_i_t(meta, i, t, out_i, t - t_window, out_x, out_z, use_lag_feat)


def get_x_z(meta, t_offset: int, length: int, window_length: int, out_x, out_z=None, out_v=None, use_lag_feat=True):
    s = meta['s']
    v = meta['v']
    assert len(s) == len(v)
    N, _ = s.shape
    t_end = t_offset + length - window_length + 1
    out_i = 0
    for i in range(N):
        for t in range(t_offset, t_end):
            get_window_x_z_at_i_t(meta, i, t, window_length, i, out_x, out_z, use_lag_feat)
            if out_v is not None:
                out_v[out_i] = v[i]
            out_i += 1
            if t % 500 == 0:
                print('i {}/{} t {}/{}'.format(i, N, t, t_end))


def get_x_z_subsample(meta, t_offset: int, length: int, window_length: int, out_x, out_z=None, out_v=None):
    s = meta['s']
    v = meta['v']
    assert len(s) == len(v)
    N, _ = s.shape
    count_x, _, _ = out_x.shape
    count_z, _, _ = out_z.shape
    count_v, = out_v.shape
    assert count_x == count_z
    assert count_x == count_v

    T = np.array(range(t_offset, t_offset + length - window_length + 1))
    p = np.array([x * 3 / len(T) for x in range(len(T))])
    p = p / np.sum(p)
    for c in range(count_x):
        i = randint(0, len(s) - 1)
        t = np.random.choice(T, p=p)
        get_window_x_z_at_i_t(meta, i, t, window_length, c, out_x, out_z)
        if out_v is not None:
            out_v[c] = v[i]
        if c % 1000 == 0:
            print('sampling {}/{}'.format(c, count_x))


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
        'g': 'd',
        'datetime_offset': datetime_offset,
        'items': items,
        'shops': shops
    }
    return meta


def load_kaggle():
    meta = get_kaggle_series()

    s = meta['s']

    N, T = s.shape

    enc_len = 180
    dec_len = 90

    # first t of the series
    t1 = 1
    train_len = T - 1 - t1

    # first t of prediction range
    t0 = t1 + train_len

    v = 1 + np.mean(s[:, t1:t0], axis=1)
    meta['v'] = v

    n_samples = 100
    n_feature = 81
    x_train = np.zeros(shape=(n_samples, enc_len + dec_len, n_feature))
    z_train = np.zeros(shape=(n_samples, enc_len + dec_len, 1))
    v_train = np.zeros(shape=(n_samples,))
    get_x_z_subsample(
        meta,
        t_offset=t1,
        length=train_len,
        window_length=enc_len + dec_len,
        out_x=x_train,
        out_z=z_train,
        out_v=v_train
    )

    v_train = np.expand_dims(v_train, axis=-1)

    # enc_x, enc_z, _ = get_x_z(
    #     meta,
    #     v,
    #     datetime_offset,
    #     t_offset=t_enc,
    #     length=enc_len,
    #     window_length=enc_len,
    #     gran=gran
    # )
    #
    # dec_x, dec_z, _ = get_x_z(
    #     meta,
    #     v,
    #     datetime_offset,
    #     t_offset=t0,
    #     length=dec_len,
    #     window_length=dec_len,
    #     gran=gran
    # )

    print('loading test_enc_x & test_enc_z')
    n_samples, _ = s.shape
    test_enc_x = np.zeros(shape=(n_samples, enc_len, n_feature))
    test_enc_z = np.zeros(shape=(n_samples, enc_len, 1))
    get_x_z(
        meta,
        t_offset=T - enc_len,
        length=enc_len,
        window_length=enc_len,
        out_x=test_enc_x,
        out_z=test_enc_z
    )

    print('loading test_dec_x')
    test_dec_x = np.zeros(shape=(n_samples, dec_len, n_feature - 13))
    get_x_z(
        meta,
        t_offset=T,
        length=dec_len,
        window_length=dec_len,
        out_x=test_dec_x,
        use_lag_feat=False
    )

    v = np.expand_dims(v, axis=-1)
    v = np.expand_dims(v, axis=-1)

    data = {
        'enc_len': enc_len,
        'dec_len': dec_len,
        'x': x_train,
        'z': z_train,
        'v': v_train,
        # 'p': p,
        # 'enc_x': enc_x,
        # 'enc_z': enc_z,
        # 'dec_x': dec_x[:, :, 13:],
        # 'dec_z': dec_z,
        'dec_v': v,
        'test_enc_x': test_enc_x,
        'test_enc_z': test_enc_z,
        'test_dec_x': test_dec_x
    }

    return meta['datetime_offset'], data


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
