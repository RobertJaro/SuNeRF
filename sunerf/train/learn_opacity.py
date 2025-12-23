import argparse

import numpy as np
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn

from sunerf.model.model import SirenNet

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Prepare opacity data')
    parser.add_argument('--data_file', type=str, default='data', help='data file')
    parser.add_argument('--out_file', type=str, default='data', help='output file')

    args = parser.parse_args()

    file = args.data_file
    out_file = args.out_file

    # load data
    data = np.loadtxt(file, skiprows=2)
    logR = np.arange(-8, 2.0, 0.5)
    logT = data[:, 0]
    log_opacity = data[:, 1:]

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # cax = ax.imshow(opacity, cmap='hot', aspect='auto', origin='lower')
    # ax.set_xlabel('logR')
    # ax.set_ylabel('logT')
    # ax.set_xticks(np.arange(len(logR)))
    # ax.set_xticklabels(logR)
    # ax.set_yticks(np.arange(len(logT)))
    # ax.set_yticklabels(logT)
    # fig.colorbar(cax)
    # plt.show()

    cm_per_solRad = (1 * u.solRad).to_value(u.cm)
    g_per_electron = 9.1093837e-28
    normalization = np.sqrt(1e4 / 1.29e-24 / cm_per_solRad)  # taken from AIA normalization

    # R = rho / T^3 * 10^18
    # rho = [g/cm^3] --> [1/cm^3] --> [drho]
    logR_normalized = logR - np.log10(normalization * g_per_electron)
    print(f'Normalized logR range: {logR_normalized.min()} - {logR_normalized.max()}')
    # kappa = [cm^2 g^-1] --> [1 / drho / ds]
    log_opacity_normalized = log_opacity + np.log10(normalization * g_per_electron) + np.log10(cm_per_solRad)

    ####################### fit neural network to data #######################
    logR_arr, logT_arr = np.meshgrid(logR_normalized, logT)
    logR_tensor = torch.tensor(logR_arr.flatten()).float()
    logT_tensor = torch.tensor(logT_arr.flatten()).float()
    opacity_tensor = torch.tensor(log_opacity_normalized.flatten()).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SirenNet(2, 1, dim=16, n_layers=3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logR_tensor = logR_tensor.to(device)
    logT_tensor = logT_tensor.to(device)
    input_tensor = torch.stack([logR_tensor, logT_tensor], dim=-1)
    opacity_tensor = opacity_tensor.to(device)[:, None]

    img_norm = Normalize(log_opacity_normalized.min(), log_opacity_normalized.max())

    for i in range(int(1e5)):
        model.train()
        optimizer.zero_grad()

        output = model(input_tensor)
        loss = criterion(output, opacity_tensor)
        loss.backward()
        optimizer.step()

        # log results
        if i % 1e4 == 0:
            print(f'Epoch {i}: Loss {loss.item()}')
            with torch.no_grad():
                model.eval()
                test_output = model(input_tensor)
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))

                im = axs[0].imshow(log_opacity_normalized, cmap='hot', aspect='auto', origin='lower', norm=img_norm)
                axs[0].set_title('Data')

                test_opacity = test_output.cpu().numpy().reshape(logR_arr.shape)
                axs[1].imshow(test_opacity, cmap='hot', aspect='auto', origin='lower', norm=img_norm)
                axs[1].set_title('Model')

                for ax in axs:
                    ax.set_xlabel('logR')
                    ax.set_ylabel('logT')
                    ax.set_xticks(np.arange(len(logR_normalized)))
                    ax.set_xticklabels(logR_normalized)
                    ax.set_yticks(np.arange(len(logT)))
                    ax.set_yticklabels(logT)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    fig.colorbar(im, cax=cax)

                fig.tight_layout()
                plt.savefig(out_file.replace('.pt', '.jpg'))
                plt.close()
            torch.save(model, args.out_file)
    torch.save(model, args.out_file)
