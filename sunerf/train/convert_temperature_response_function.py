import argparse
import os

import numpy as np


def convert_response_function(response_file, channels=None, log_T_range=None, normalization=None):
    log_T_range = log_T_range if log_T_range is not None else np.arange(4, 9.001, 0.005).astype(np.float32)
    temperature_response_function = np.load(response_file)

    temperature = temperature_response_function['temperature']
    response = {k: value for k, value in temperature_response_function.items() if k != 'temperature'}

    if channels is not None:
        channels = args.channels
    else:
        channels = list(response.keys())

    response = np.stack([response[c] for c in channels], 0)
    # normalize data
    normalization = np.max(response) if normalization is None else normalization
    response = response / normalization
    temperature_interpolated = 10 ** log_T_range
    response_interpolated = np.stack(
        [np.interp(temperature_interpolated, temperature, response[i]) for i in range(response.shape[0])], 0)

    return temperature_interpolated, response_interpolated, normalization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--channels', type=int, nargs='+', required=False, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    temperature_interpolated, response_interpolated, normalization = convert_response_function(args.response_file,
                                                                                               args.channels)

    np.savez(args.out_file, temperature=np.log10(temperature_interpolated), response=np.log10(response_interpolated),
             normalization=np.log10(normalization))
