import glob

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    files = ['']
    files += ['/glade/work/rjarolim/sunerf-cme-v2/3view_v01/evaluation/center_of_mass.npz']
    labels = ['all', 'ecliptic', '3 views']

    diffs = []
    mass_diffs = []
    velocity_com_diffs = []
    velocity_sf_diffs = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        #
        center_of_mass_pred = data['center_of_mass_pred']
        center_of_mass_true = data['center_of_mass_true']
        diff = np.abs(center_of_mass_true - center_of_mass_pred).mean(0)
        diffs.append(diff)
        #
        mass_diffs.append(data['mass_relative_diff'])
        #
        velocity_com_diff = (data['velocity_com_pred'] - data['velocity_com_true']) / data['velocity_com_true'] * 100
        velocity_com_diffs.append(velocity_com_diff)
        #
        velocity_sf_diff = (data['velocity_sf_pred'] - data['velocity_sf_true']) / data['velocity_sf_true'] * 100
        velocity_sf_diffs.append(velocity_sf_diff)


    diffs = np.array(diffs)
    mass_diffs = np.array(mass_diffs)
    velocity_com_diffs = np.array(velocity_com_diffs)
    velocity_sf_diffs = np.array(velocity_sf_diffs)


    angles = [(a - 360) if a > 180 else a for a in angles ]

    fig, axs = plt.subplots(1, 5, figsize=(12, 2))

    ax = axs[0]
    l1 = ax.scatter(angles, velocity_com_diffs[:], label='side')
    l2 = ax.scatter(angles[0], velocity_com_diffs[0], label='halo')
    l3 = ax.scatter(angles[3:6], velocity_com_diffs[3:6], label='>100 deg')
    # l4 = ax.scatter(angles[-2], velocity_com_diffs[-2], label='polar')
    l4 = ax.scatter(angles[-1], velocity_com_diffs[-1], label='3 views')
    ax.set_xlabel('Observer [deg]')
    ax.set_ylabel(r'$\Delta v_\text{CoM}$ [%]')
    ax.axhline(0, color='k', linestyle='--', lw=0.5)

    ax = axs[1]
    ax.scatter(angles, velocity_sf_diffs[:], label='side')
    ax.scatter(angles[0], velocity_sf_diffs[0], label='halo')
    ax.scatter(angles[3:6], velocity_sf_diffs[3:6], label='>100 deg')
    # ax.scatter(angles[-2], velocity_sf_diffs[-2], label='polar')
    ax.scatter(angles[-1], velocity_sf_diffs[-1], label='3 views')
    ax.set_xlabel('Observer [deg]')
    ax.set_ylabel(r'$\Delta v_\text{FRT}$ [%]')
    ax.axhline(0, color='k', linestyle='--', lw=0.5)

    ax = axs[2]
    ax.scatter(angles, diffs[:, 1], label='side')
    ax.scatter(angles[0], diffs[0, 1], label='halo')
    ax.scatter(angles[3:6], diffs[3:6, 1], label='>100 deg')
    # ax.scatter(angles[-2], diffs[-2, 1], label='polar')
    ax.scatter(angles[-1], diffs[-1, 1], label='3 views')
    ax.set_xlabel('Observer [deg]')
    ax.set_ylabel(r'$\Delta\theta$ [deg]')

    ax = axs[3]
    ax.scatter(angles, diffs[:, 2], label='side')
    ax.scatter(angles[0], diffs[0, 2], label='halo')
    ax.scatter(angles[3:6], diffs[3:6, 2], label='>100 deg')
    # ax.scatter(angles[-2], diffs[-2, 2], label='polar')
    ax.scatter(angles[-1], diffs[-1, 2], label='3 views')
    ax.set_xlabel('Observer [deg]')
    ax.set_ylabel('$\Delta\phi$ [deg]')

    ax = axs[4]
    ax.scatter(angles, mass_diffs[:], label='side')
    ax.scatter(angles[0], mass_diffs[0], label='halo')
    ax.scatter(angles[3:6], mass_diffs[3:6], label='>100 deg')
    # ax.scatter(angles[-2], mass_diffs[-2], label='polar')
    ax.scatter(angles[-1], mass_diffs[-1], label='3 views')
    ax.set_xlabel('Observer [deg]')
    ax.set_ylabel(r'$\Delta M_\text{CME}$ [%]')

    # add legend axis
    fig.legend(loc='outside upper center', handles=[l1, l2, l3, l4], ncol=5, fontsize=8, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig('/glade/work/rjarolim/sunerf-cme-v2/variations/angle_comparison.png', dpi=300)
    plt.close(fig)

    central_indices = [0, 1, 2, 6, 7, 8]
    print('Mean diffs:')
    print('Velocity CoM:', velocity_com_diffs[central_indices].mean(0))
    print('Velocity FRT:', velocity_sf_diffs[central_indices].mean(0))
    print('Mass:', mass_diffs[central_indices].mean(0))
    print('Theta:', diffs[central_indices, 1].mean(0))
    print('Phi:', diffs[central_indices, 2].mean(0))