import argparse

import numpy as np
from astropy.io.fits import getdata
from matplotlib import pyplot as plt


def main(file_174, file_304, out_path, target_ne):
    eui_174 = getdata(file_174)
    eui_304 = getdata(file_304)

    ne = 10 ** np.linspace(8, 18, eui_174.shape[0])
    temperature = 10 ** np.linspace(4, 8, eui_174.shape[1])

    eui_174_G = eui_174 / ne[:, None] ** 2 * 1e10
    eui_304_G = eui_304 / ne[:, None] ** 2 * 1e10

    ne_idx = np.argmin(np.abs(ne - target_ne))
    eui_174_response = eui_174_G[ne_idx]
    eui_304_response = eui_304_G[ne_idx]

    temperature_response = {'fsi_174': eui_174_response, 'fsi_304': eui_304_response}

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
    parser = argparse.ArgumentParser(description='Plot and save EUI response functions.')
    parser.add_argument('--file_174', type=str, required=True, help='Path to the 174 FITS file')
    parser.add_argument('--file_304', type=str, required=True, help='Path to the 304 FITS file')
    parser.add_argument('--out_path', type=str, required=True, help='Output directory')
    parser.add_argument('--target_ne', type=float, default=1e9, help='Target electron density')
    args = parser.parse_args()

    main(args.file_174, args.file_304, args.out_path, args.target_ne)
