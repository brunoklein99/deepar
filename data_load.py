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
def load_parts():
    df = pd.read_csv('data/carparts.csv')

    # "out of 2509 series with complete records for 51 months"
    df = df.dropna()

    parts = df.values
    parts = parts[:, 1:]

    indexes = get_keep_indexes(parts)
    parts = parts[indexes]

    # x and y are the same time series, with y shifted by 1
    x = parts[:, :50]
    y = parts[:, 1:50]

    return x, y
