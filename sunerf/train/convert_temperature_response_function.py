import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--channels', type=int, nargs='+', required=False, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    temperature_response_function = np.load(args.response_file)
    temperature = temperature_response_function['temperature']
    response = {k: value for k, value in temperature_response_function.items() if k != 'temperature'}
    if args.channels is not None:
        channels = args.channels
    else:
        channels = list(response.keys())
    print(f'Channels: {channels}')
    response = np.stack([temperature_response_function[c] for c in channels], 0)

    # normalize data
    normalization = 1e-24 #np.max(response)
    print(f'Normalization: {normalization:.2e}')
    response = response / normalization

    x = np.logspace(4, 9, 101, dtype=np.float32)
    response_interpolated = np.stack([np.interp(x, temperature, response[i]) for i in range(response.shape[0])], 0)

    n_channels = response.shape[0]
    fig, axs = plt.subplots(n_channels, 1, figsize=(10, 5 * n_channels))

    for i, ax in enumerate(axs):
        ax.plot(temperature, response[i], 'o', color='black')
        ax.plot(x, response_interpolated[i], color='red')
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.savefig(args.out_file.replace('.npz', '.png'))
    plt.close()

    np.savez(args.out_file, temperature=np.log10(x), response=np.log10(response_interpolated),
             normalization=np.log10(normalization))
