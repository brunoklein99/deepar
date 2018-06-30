import json
from data_load import get_parts_series


def write_file(filename, datetime_offset, s):
    with open(filename, 'w') as f:
        for i in range(len(s)):
            f.write(json.dumps({
                'start': str(datetime_offset),
                'target': list(s[i])
            }) + '\n')


if __name__ == '__main__':
    datetime_offset, s = get_parts_series()

    write_file('data/sagemaker_train.json', datetime_offset, s[:, :-8])
    write_file('data/sagemaker_valid.json', datetime_offset, s)

    print('finished')
