import json
import numpy as np

from data_load import get_parts_series

if __name__ == '__main__':
    with open('data/output.json') as f:
        _, s = get_parts_series()
        true = s[:, -8:]
        j = json.load(f)
        j = j['predictions']
        pred = []
        for i in range(len(j)):
            pred.append(j[i]['mean'])
        pred = np.array(pred)
        print('rmse {}'.format(np.sqrt(np.mean(np.square(pred - true)))))
