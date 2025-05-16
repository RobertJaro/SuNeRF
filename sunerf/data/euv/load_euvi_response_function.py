import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav


def main(response_file, out_path, mode):
    data = readsav(response_file)

    temperature = data['p0'][0][5]  # (61, )

    channel_keys = data['p0'][0][8][:, 0]

    modes = data['p0'][0][9][0, :]
    mode_idx = np.where(modes == mode)[0][0]
    # print('Modes', modes, f"select --> {data['p0'][0][9][0, mode_idx]}")

    # 6 = electrons; 7 = photons
    # response units = 1e44 EM
    response = data['p0'][0][7]  # 4, 4, 61
    response = response[:, mode_idx, :]  # use S1 default ['OPEN', 'S1', 'S2', 'DBL']

    temperature_response = {str(k): response[i] for i, k in enumerate(channel_keys)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key, resp in temperature_response.items():
        ax.plot(temperature, resp, label=key)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Response')
    ax.set_title('Response Functions')
    ax.legend()
    plt.savefig(out_path.replace('.npz', '.png'))
    plt.show()

    np.savez(out_path, temperature=temperature, **temperature_response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process EUVI response function.')
    parser.add_argument('--response_file', type=str, required=True, help='Path to the response .geny file')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save the output .npz file')
    parser.add_argument('--mode', type=str, default='S1', help='Mode to select (e.g., S1, S2, etc.)')

    args = parser.parse_args()
    main(args.response_file, args.out_path, args.mode.encode())
