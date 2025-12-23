import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import readsav


def main(euv_resp_path, out_path):
    euv_resp = readsav(euv_resp_path)

    temperature = 10 ** euv_resp['aiaresp'][0][7]
    response = euv_resp['aiaresp'][0][8]
    channel_keys = euv_resp['aiaresp'][0][5].astype(str)

    temperature_response = {k: response[i] for i, k in enumerate(channel_keys)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key, response in temperature_response.items():
        ax.plot(temperature, response, label=key)

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
    parser = argparse.ArgumentParser(description='Plot and save AIA response functions.')
    parser.add_argument('--response_file', type=str, required=True, help='Path to aia_euv_resp.sav file')
    parser.add_argument('--out_path', type=str, required=True, help='Output path for .npz file')
    args = parser.parse_args()
    main(args.response_file, args.out_path)
