import argparse
import datetime
import os

import numpy as np
import torch
import xarray as xr
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torch.nn import DataParallel

from sunerf.model.water import WaterVaporModel
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.water import WaterRadiativeTransfer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--work_directory', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--reload_data', action='store_true')
    args = parser.parse_args()

    meters_per_ds = 1e4

    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    model_path = os.path.join(out_path, 'save_state.nef')
    dataset_path = os.path.join(out_path, 'dataset.pt')

    file_path = os.path.join(args.data_path, 'qvapor_test.nc')
    z_file_path = os.path.join(args.data_path, 'z_test.nc')
    p_file_path = os.path.join(args.data_path, 'p_test.nc')

    # mixing ratio of water
    qvapor_data = xr.open_dataset(file_path)
    z_data = xr.open_dataset(z_file_path)
    p_data = xr.open_dataset(p_file_path)

    # z = z_data['Z'].values.T  # in meters
    z = np.linspace(0, 1, z_data['Z'].shape[0]) * 15000  # in meters, assuming a fixed height for simplicity
    z = np.tile(z[None, :], (z_data['Z'].shape[1], 1))  # repeat for each longitude
    z = (z[:, 1:] + z[:, :-1]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(z.T, origin='lower', cmap='viridis', aspect='auto')
    ax.set_title('Height (Z)')
    fig.colorbar(im, ax=ax, label='Height (m)')
    plt.savefig(os.path.join(out_path, 'height_z.jpg'))
    plt.close(fig)

    # water vapor density
    # rho_water = mixing ratio * rho_air = mixing ratio * p / (R * T) = C * mixing ratio * p
    rho_true_npy = qvapor_data['QVAPOR'].values.T * p_data['P'].values.T  # in kg/m^3
    rho_true_npy = np.log10(rho_true_npy)  # convert to log scale
    print('SHAPE', rho_true_npy.shape)
    # rho_true_npy[1:] = rho_true_npy[0:1]

    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #
    # ax.plot((10 **rho_true_npy).mean(0), z[0], label='Mean')
    # ax.plot((10 ** rho_true_npy).std(0), z[0], label='Std')
    #
    # ax.set_xlabel('Log10 Water Vapor Density (kg/m^3)')
    # ax.set_ylabel('Height (m)')
    # ax.set_title('Water Vapor Density Mean and Std')
    #
    # # plt.semilogx()
    #
    # fig.savefig(os.path.join(out_path, 'rho_true_mean_std.jpg'))
    # plt.close(fig)

    model = WaterVaporModel()
    parallel_model = DataParallel(model)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parallel_model.train()
    parallel_model.to(device)

    longitude = z_data['XLONG'].values
    x = (1 * u.R_earth).to_value(u.m) * np.cos(np.deg2rad(longitude))
    x = x - x.min()

    coords_npy = np.zeros((*rho_true_npy.shape, 4), dtype=np.float32)
    coords_npy[..., 2] = z
    coords_npy[..., 0] = x[:, None]
    coords_npy = coords_npy / meters_per_ds  # convert to model units

    # visualize coordinates
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))

    for i in range(4):
        ax = axs[i]
        im = ax.imshow(coords_npy[..., i].T, origin='lower', cmap='viridis', aspect='auto')
        ax.set_title(f'Coordinate {i}')
        fig.colorbar(im, ax=ax, label=f'Coordinate {i}')
    plt.savefig(os.path.join(out_path, 'coordinates.jpg'))
    plt.close(fig)

    valid_coords = torch.tensor(coords_npy[::10], dtype=torch.float32)
    valid_rho_true = torch.tensor(rho_true_npy[::10], dtype=torch.float32)

    coords = torch.tensor(coords_npy, dtype=torch.float32).reshape(-1, 4)  # shape (x*y*z*t, 4)
    rho_true = torch.tensor(rho_true_npy, dtype=torch.float32).reshape(-1, 1)

    batch_size = int(2 ** 16)

    # dummy module for rendering
    rendering_module = WaterRadiativeTransfer()
    rendering = BasicRenderingModule(model=model,
                                     rendering_modules={'QVAPOR': rendering_module},
                                     sampling_config={'type': 'flat_earth'})
    rendering.to(device)


    def save_model():
        state = {
            # sunerf  rendering module
            'rendering': rendering,
            # data infor
            'data_config': {},
            # data scaling
            'meters_per_ds': meters_per_ds,
            'seconds_per_dt': 1,
            'ref_date': datetime.datetime(2025, 1, 1),  # The Earf was created in 2025
        }
        torch.save(state, model_path)

    ########################################################################################################################
    # start training
    epochs = 10000
    for epoch in range(epochs):
        total_loss = []
        # shuffle data
        indices = torch.randperm(coords.shape[0])
        coords = coords[indices]
        rho_true = rho_true[indices]
        for i in range(np.ceil(len(coords) / batch_size).astype(int)):
            optimizer.zero_grad()
            batch_rho_true = rho_true[i * batch_size: (i + 1) * batch_size]
            batch_coords = coords[i * batch_size: (i + 1) * batch_size]
            # move to device
            batch_rho_true = batch_rho_true.to(device)
            batch_coords = batch_coords.to(device)
            #
            model_out = parallel_model(batch_coords)
            batch_rho_pred = model_out['log10_rho']
            loss = (batch_rho_pred - batch_rho_true) ** 2
            loss = loss.mean()
            assert not torch.isnan(loss).any(), 'NaN detected in loss. Stopping training.'
            loss.backward()
            optimizer.step()
            total_loss.append(loss.cpu().detach().numpy())
        print(f'Epoch {epoch}, loss: {np.mean(total_loss):.6f}')

        # validation every 100 epochs
        if not (epoch + 1) % 100 == 0 and epoch > 0:
            continue

        ########################## Validation
        with torch.no_grad():
            parallel_model.eval()

            valid_rho_pred = parallel_model(valid_coords)['log10_rho']
            valid_rho_pred = valid_rho_pred[..., 0]

            rho_norm = Normalize()
            # plot validation image
            fig, axs = plt.subplots(2, 1, figsize=(10, 5))

            ax = axs[0]
            im = ax.imshow(valid_rho_true.cpu().numpy().T, origin='lower', cmap='jet', norm=rho_norm, aspect='auto')
            ax.set_title('rho_true')
            fig.colorbar(im, ax=ax)

            ax = axs[1]
            im = ax.imshow(valid_rho_pred.cpu().numpy().T, origin='lower', cmap='jet', norm=rho_norm, aspect='auto')
            ax.set_title('rho_pred')
            fig.colorbar(im, ax=ax)

            fig.tight_layout()
            plt.savefig(os.path.join(out_path, f'validation_{epoch + 1:06d}.jpg'))
            plt.close()

            ###########################################################################################################################
            # plot validation sum
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            ax = axs[0]
            im = ax.imshow((10 ** valid_rho_true).sum(1, keepdim=True).cpu().numpy().T, origin='lower', cmap='viridis', aspect='auto')
            ax.set_title('rho_true sum')
            fig.colorbar(im, ax=ax)

            ax = axs[1]
            im = ax.imshow((10 ** valid_rho_pred).sum(1, keepdim=True).cpu().numpy().T, origin='lower', cmap='viridis', aspect='auto')
            ax.set_title('rho_pred sum')
            fig.colorbar(im, ax=ax)

            plt.savefig(os.path.join(out_path, f'validation_sum_{epoch + 1:06d}.jpg'))
            plt.close(fig)

            #########################################################################################################################
            # validation image generation
            obs_angle = 45 * u.deg
            resolution = 128
            angles = np.linspace(-obs_angle.to_value(u.rad) / 2, obs_angle.to_value(u.rad) / 2, resolution)
            rays_d = np.stack([np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=-1)  # (128, 3)
            rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)  # normalize directions

            z_obs = 15e3 * u.m
            x_buffer = z_obs / np.cos(obs_angle.to_value(u.rad) / 2)
            x_obs = np.linspace(x_buffer, 73785.75 * u.m - x_buffer, 256)
            images = []
            for x in x_obs:
                rays_o = np.array([x.to_value(u.m), 0, z_obs.to_value(u.m)]) / meters_per_ds  # (3,)
                rays_o = np.tile(rays_o, (resolution, 1))  # (128, 3)

                rays_o = torch.tensor(rays_o, dtype=torch.float32, device=device)  # (128, 3)
                rays_d = torch.tensor(rays_d, dtype=torch.float32, device=device)  # (128, 3)
                time = torch.zeros_like(rays_o[..., 0:1], dtype=torch.float32, device=device)
                rays = torch.stack([rays_o, rays_d], 1)
                batch = {'QVAPOR': {'rays': rays, 'time': time, 'instrument': 'QVAPOR'}}
                rendering_out = rendering(batch)
                images.append(rendering_out['model_out']['QVAPOR']['image'].cpu().numpy())

            images = np.stack(images, axis=0)  # (256, 128, 1)

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            im = ax.imshow(images, origin='lower', norm='log')
            ax.set_xlabel('Scan (deg)')
            ax.set_ylabel('Observer position (m)')

            ax.set_title('Water vapor brightness')
            fig.colorbar(im, ax=ax, label='Brightness (dB)')
            fig.tight_layout()
            plt.savefig(os.path.join(out_path, f'validation_image_{epoch + 1:06d}.jpg'))
            plt.close(fig)

        parallel_model.train()

        ########################## Save model
        save_model()
