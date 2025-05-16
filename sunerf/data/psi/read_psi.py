import os

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.colors import LogNorm

from sunerf.train.coordinate_transformation import spherical_to_cartesian


def read_PSI(rho_hdf5, T_hdf5, min_radius=1.0, max_radius=2.6):
    print(f'Reading {rho_hdf5} and {T_hdf5}')
    time = int(os.path.basename(rho_hdf5)[3:9]) - 1813
    with File(rho_hdf5, 'r') as h5file:
        # get coordinate information
        r_mhd = np.array(h5file['dim1'], dtype=np.float32)
        th_mhd = np.array(h5file['dim2'], dtype=np.float32) - np.pi / 2
        phi_mhd = np.array(h5file['dim3'], dtype=np.float32)

        rho = np.array(h5file['Data'], dtype=np.float32).T
    with File(T_hdf5, 'r') as h5file:
        T = np.array(h5file['Data'], dtype=np.float32).T

    print(f'r_range: {r_mhd.min(), r_mhd.max()}, th_range: {th_mhd.min(), th_mhd.max()}, phi_range: {phi_mhd.min(), phi_mhd.max()}')
    # remove values outside of the max_radius
    r_mask = (r_mhd <= max_radius)
    if min_radius is not None:
        r_mask = r_mask & (r_mhd > min_radius)
    # apply mask
    r_mhd = r_mhd[r_mask]
    rho = rho[r_mask]
    T = T[r_mask]

    # create spherical coordinates
    spherical_coordinates = np.stack(np.meshgrid(r_mhd, th_mhd, phi_mhd, indexing='ij'), axis=-1)
    # create cartesian coordinates
    cartesian_coordinates = spherical_to_cartesian(spherical_coordinates)

    T[T < 0] = np.nan
    rho[rho < 0] = np.nan

    # normalize values
    T = T * 2.807066716734894e7
    rho = rho * 1.0e8

    return {'rho': rho, 'T': T, 'time': time,
            'spherical_coordinates': spherical_coordinates, 'cartesian_coordinates': cartesian_coordinates}

if __name__ == '__main__':
    rho_hdf5 = '/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd/rho/rho001813.h5'
    T_hdf5 = '/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd/t/t001813.h5'
    psi_data = read_PSI(rho_hdf5, T_hdf5)
    print(f'rho shape: {psi_data["rho"].shape}, T shape: {psi_data["T"].shape}, spherical_coordinates shape: {psi_data["spherical_coordinates"].shape}, time: {psi_data["time"]}')

    ############################################################################################################
    # lon slice
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im = axs[0].imshow(psi_data['rho'][:, :, 0], norm=LogNorm(), origin='lower')
    axs[0].set_title('rho')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(psi_data['T'][:, :, 0], norm=LogNorm(), origin='lower')
    axs[1].set_title('T')
    fig.colorbar(im, ax=axs[1])

    fig.savefig('/glade/work/rjarolim/sunerf/psi_cube/lon_slice.jpg')
    plt.close(fig)

    ############################################################################################################
    # integrated radius
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im = axs[0].imshow(psi_data['rho'].mean(0), norm=LogNorm(), origin='lower')
    axs[0].set_title('rho')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(psi_data['T'].mean(0), norm=LogNorm(), origin='lower')
    axs[1].set_title('T')
    fig.colorbar(im, ax=axs[1])

    fig.savefig('/glade/work/rjarolim/sunerf/psi_cube/radial.jpg')
    plt.close(fig)

    ############################################################################################################
    # lat slice
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im = axs[0].imshow(psi_data['rho'][:, 100, :], norm=LogNorm(), origin='lower')
    axs[0].set_title('rho')
    fig.colorbar(im, ax=axs[0])

    im = axs[1].imshow(psi_data['T'][:, 100, :], norm=LogNorm(), origin='lower')
    axs[1].set_title('T')
    fig.colorbar(im, ax=axs[1])

    fig.savefig('/glade/work/rjarolim/sunerf/psi_cube/lat_slice.jpg')
    plt.close(fig)