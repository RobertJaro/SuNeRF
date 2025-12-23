import os

import numpy as np
from matplotlib import pyplot as plt

def sigma_to_snr(sigma):
    print(sigma)
    return 1.0 / sigma if sigma != 0 else '-'

def snr_to_sigma(snr):
    return 1.0 / snr

if __name__ == '__main__':
    out_path = '/glade/work/rjarolim/sunerf-cme-v2/noise/evaluation'

    os.makedirs(out_path, exist_ok=True)

    files = ['/glade/work/rjarolim/sunerf-cme-v2/helio_v04/evaluation_full_v02/center_of_mass.npz',
             '/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.01_v01/evaluation/center_of_mass.npz',
             '/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.02_v01/evaluation/center_of_mass.npz',
             '/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.05_v01/evaluation/center_of_mass.npz',
             '/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.10_v01/evaluation/center_of_mass.npz',
             '/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.20_v01/evaluation/center_of_mass.npz',
             '/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.30_v01/evaluation/center_of_mass.npz'
             ]

    noise = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

    maes = []
    correlation_coeffs = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        #
        c = np.mean(data['corr_coeff'])
        dc = np.std(data['corr_coeff'])
        correlation_coeffs.append((c, dc))
        #
        mae = np.mean(data['mae'])
        dmae = np.std(data['mae'])
        maes.append((mae, dmae))

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    ax = axs[0]
    ax.errorbar(noise, [m[0] for m in maes], yerr=[m[1] for m in maes], marker='o', capsize=5)
    ax.set_xlabel('Noise Level $\sigma$')
    ax.set_ylabel('MAE [N$_e$ cm$^{-3}$]')

    ax = axs[1]
    ax.errorbar(noise,
                [c[0] for c in correlation_coeffs],
                yerr=[c[1] for c in correlation_coeffs], marker='o', capsize=5)
    ax.set_xlabel('Noise Level $\sigma$')
    ax.set_ylabel('Correlation Coefficient (CC)')

    for ax in axs:
        secax = ax.secondary_xaxis("top")
        secax.set_xlabel(r'$\overline{\mathrm{SNR}}$')
        ticks = ax.get_xticks()
        labels = ["∞" if t == 0 else f"{1 / t:.1f}" for t in ticks]
        secax.set_xticks(ticks)
        secax.set_xticklabels(labels)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'noise_comparison.png'), dpi=300)
    plt.close()