import json
from data_load import get_parts_series

if __name__ == '__main__':

    datetime_offset, s = get_parts_series()
    s = s[:, :-8]

    with open('data/sagemaker_train.json', 'w') as f:
        for i in range(len(s)):
            f.write(json.dumps({
                'start': str(datetime_offset),
                'target': list(s[i])
            })+'\n')

    print('finished')
