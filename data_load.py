import pandas as pd
import numpy as np


def get_indexes_with_n_positive(arr, n=10):
    b = np.array(arr, dtype=np.bool)
    b = np.sum(b, axis=-1)
    return np.squeeze(np.argwhere(b >= n))


# http://isiarticles.com/bundles/Article/pre/pdf/20710.pdf (reference from DeepAR)
# end of page 1, right below abstract
def load_parts():
    df = pd.read_csv('data/carparts.csv')

    # "out of 2509 series with complete records for 51 months"
    df = df.dropna()

    parts = df.values

    # "ten or more months with positive demands"
    indexes = get_indexes_with_n_positive(parts, n=10)
    parts = parts[indexes]

    print(len(parts))

    # "at least some positive demands in the first 15"
    indexes = get_indexes_with_n_positive(parts[:, :15], n=1)
    parts = parts[indexes]

    print(len(parts))

    # "... and the last 15 months"
    indexes = get_indexes_with_n_positive(parts[:, -15:], n=1)
    parts = parts[indexes]

    # this does not yield the 1046 time series they state in the paper
    print(len(parts))

    parts = parts[:, 1:]

    # x and y are the same timeseries, with y shifted by 1
    x = parts[:, :50]
    y = parts[:, 1:50]

    return x, y
