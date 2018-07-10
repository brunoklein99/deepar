import json

from dateutil.relativedelta import relativedelta

from data_load import get_parts_series, get_elec_series, get_elem_series


def write_inference(filename, datetime_offset, s):
    instances = []

    for i in range(len(s)):
        instances.append({
            'start': str(datetime_offset),
            'target': list(s[i])
        })

    configuration = {
        "num_samples": 50,
        "output_types": ["mean", "quantiles", "samples"],
        "quantiles": ["0.5", "0.9"]
    }

    with open(filename, 'w') as f:
        f.write(json.dumps({
            'instances': instances,
            'configuration': configuration
        }))


def write_file(filename, datetime_offset, s):
    with open(filename, 'w') as f:
        for i in range(len(s)):
            f.write(json.dumps({
                'start': str(datetime_offset),
                'target': list(s[i])
            }) + '\n')


if __name__ == '__main__':
    offset_train, s = get_elem_series()

    _, T = s.shape

    enc_len = 217
    dec_len = 31

    offset_valid = offset_train + relativedelta(days=T - dec_len - enc_len)

    write_file('data/sagemaker_train.json', offset_train, s[:, :-dec_len])
    write_file('data/sagemaker_valid.json', offset_train, s)
    write_inference('data/inference.json', offset_valid, s[:, -(enc_len + dec_len):-dec_len])

    print('finished')
