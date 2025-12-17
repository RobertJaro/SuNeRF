import copy
from typing import Optional

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from skimage.metrics import structural_similarity
from sklearn.linear_model import LinearRegression

from sunerf.data.date_util import unnormalize_datetime
from sunerf.data.utils import sdo_img_norm


class BaseCallback(Callback):

    def __init__(self, ds_key, name=None):
        super().__init__()
        self.ds_key = ds_key
        self.name = name if name is not None else ds_key

    def get_validation_outputs(self, pl_module):
        if self.ds_key not in pl_module.validation_outputs:
            return None
        outputs = pl_module.validation_outputs[self.ds_key]
        return outputs


class AbsorptionCallback(BaseCallback):

    def __init__(self, ds_key, image_shape):
        super().__init__(ds_key)
        self.image_shape = image_shape

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        kappa = outputs['kappa'][..., 0]
        log_kappa = outputs['log_kappa'][..., 0]
        log_ne = outputs['log_ne'][..., 0]
        log_T = outputs['log_T'][..., 0]
        alpha = 10 ** (log_kappa + log_ne)

        extent = [log_T[0, 0], log_T[-1, -1], log_ne[0, 0], log_ne[-1, -1]]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        ax = axs[0]
        im = ax.imshow(kappa.T, cmap='viridis', norm='log', extent=extent, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'$\\nu$')
        ax.set_xlabel('log(T)')
        ax.set_ylabel('log(n_e)')

        ax = axs[1]
        im = ax.imshow(alpha.T, cmap='viridis', norm='log', extent=extent, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'$\\alpha$')
        ax.set_xlabel('log(T)')
        ax.set_ylabel('log(n_e)')

        fig.tight_layout()
        wandb.log({'absorption': fig})
        plt.close('all')


class TestImageCallback(BaseCallback):

    def __init__(self, ds_key, image_shape, cmap='gray'):
        super().__init__(ds_key)
        self.image_shape = image_shape
        self.cmap = plt.get_cmap(cmap)
        self.normalize = ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005), clip=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        fine_image = self.normalize(outputs['fine_image'])
        target_image = self.normalize(outputs['target_image'])
        coarse_image = self.normalize(outputs['coarse_image'])

        self.plot_samples(fine_image, coarse_image, outputs['height_map'], outputs['absorption_map'],
                          target_image, outputs['z_vals_stratified'], outputs['z_vals_hierarchical'],
                          outputs['distance'].mean(), self.cmap)

        val_loss = ((fine_image - target_image) ** 2).mean()
        val_ssim = structural_similarity(target_image[..., 0], fine_image[..., 0], data_range=1)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({'validation.loss': val_loss, 'validation.ssim': val_ssim, 'validation.psnr': val_psnr})

    def plot_samples(self, fine_image, coarse_image, height_map, absorption_map, target_image, z_vals_stratified,
                     z_vals_hierach, distance, cmap):
        # Log example images on wandb
        # # Plot example outputs

        fig, ax = plt.subplots(1, 6, figsize=(30, 4))

        ax[0].imshow(target_image[..., 0], cmap=cmap, norm=sdo_img_norm)
        ax[0].set_title(f'Target')
        ax[1].imshow(fine_image[..., 0], cmap=cmap, norm=sdo_img_norm)
        ax[1].set_title(f'Fine')
        ax[2].imshow(coarse_image[..., 0], cmap=cmap, norm=sdo_img_norm)
        ax[2].set_title(f'Coarse')
        ax[3].imshow(height_map, cmap='plasma', vmin=1, vmax=1.3)
        ax[3].set_title(f'Emission Height')
        ax[4].imshow(absorption_map, cmap='viridis', vmin=0)
        ax[4].set_title(f'Absorption')

        # select index
        y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4  # select point in first quadrant
        plot_ray_sampling(z_vals_stratified[y, x], z_vals_hierach[y, x], ax[-1])

        wandb.log({"Comparison": fig})
        plt.close('all')


class PlasmaImageCallback(BaseCallback):

    def __init__(self, ds_key, image_shape, cmaps=None):
        super().__init__(ds_key)
        self.image_shape = image_shape
        self.cmaps = cmaps
        self.normalize = ImageNormalize(vmin=0, vmax=1, clip=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        pred_image = outputs['pred_image']
        target_image = outputs['target_image']

        cmaps = self.cmaps
        cmaps = ['gray'] * pred_image.shape[0] if cmaps is None else cmaps

        fig, axs = plt.subplots(2, len(cmaps), figsize=(3 * len(cmaps), 6))
        for i, cmap in enumerate(cmaps):
            cmap = plt.get_cmap(cmap)
            col = axs[:, i]

            v_max = np.nanmax(target_image[..., i])
            im = col[0].imshow(target_image[..., i], cmap=cmap, vmin=0, vmax=v_max)
            divider = make_axes_locatable(col[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            col[0].set_title(f'Target')

            im = col[1].imshow(pred_image[..., i], cmap=cmap, vmin=0, vmax=v_max)
            divider = make_axes_locatable(col[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            col[1].set_title(f'Prediction')

        [ax.set_axis_off() for ax in axs.flatten()]

        fig.tight_layout()
        wandb.log({f'images.{self.ds_key}': fig})
        plt.close('all')

        self.plot_integrated_quantities(outputs['height_map'], outputs['mean_T'], outputs['total_ne'],
                                        outputs['mean_absorption'],
                                        outputs['z_vals_stratified'], outputs['z_vals_hierarchical'],
                                        outputs['distance'].mean())

        val_loss = ((pred_image - target_image) ** 2).mean()
        val_ssim = []
        for i in range(target_image.shape[-1]):
            val_ssim += [structural_similarity(target_image[..., i], pred_image[..., i], data_range=1)]
        val_ssim = np.mean(val_ssim)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({f'validation.loss.{self.ds_key}': val_loss,
                   f'validation.ssim.{self.ds_key}': val_ssim,
                   f'validation.psnr.{self.ds_key}': val_psnr})

    def plot_integrated_quantities(self, height_map, mean_T, total_ne, absorption, z_vals_stratified,
                                   z_vals_hierach, distance, ):
        fig, axs = plt.subplots(1, 5, figsize=(24, 4))

        ax = axs[0]
        im = ax.imshow(height_map, cmap='cividis', vmin=1, vmax=1.3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Emission Height')

        ax = axs[1]
        im = ax.imshow(mean_T, cmap='inferno')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'mean log(T)')

        ax = axs[2]
        im = ax.imshow(total_ne, cmap='viridis', norm='log')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Total $n_e$')

        ax = axs[3]
        im = ax.imshow(absorption, cmap='cool')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Mean Absorption')

        # select index
        y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4  # select point in first quadrant
        plot_ray_sampling(z_vals_stratified[y, x], z_vals_hierach[y, x], axs[-1])

        fig.tight_layout()
        wandb.log({f'integrated_quantities.{self.ds_key}': fig})
        plt.close('all')


class ThomsonImageCallback(BaseCallback):

    def __init__(self, ds_key, image_shape):
        super().__init__(ds_key)
        self.image_shape = image_shape
        self.normalize = ImageNormalize(vmin=0, vmax=1, clip=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        model_image = outputs['model_image']
        target_image = outputs['target_image']

        target_ratio = outputs['target_ratio']
        model_ratio = outputs['model_ratio']

        model_image[np.isnan(target_image)] = np.nan
        model_ratio[np.isnan(target_image).any(-1)] = np.nan

        fig, axs = plt.subplots(3, 2, figsize=(7, 9))

        # pB and tB images
        for i in range(2):
            ax = axs[i, 0]
            v_max = np.nanmax(target_image[..., i])
            v_min = np.nanmin(target_image[..., i])
            im = ax.imshow(target_image[..., i], cmap='plasma', vmin=v_min, vmax=v_max)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title(f'Target')

            ax = axs[i, 1]
            im = ax.imshow(model_image[..., i], cmap='plasma', vmin=v_min, vmax=v_max)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title(f'Fine')

        # ratio images
        ax = axs[2, 0]
        v_max = np.nanmax(target_ratio[..., 0])
        im = ax.imshow(target_ratio[..., 0], cmap='plasma', vmin=0, vmax=v_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Target')

        ax = axs[2, 1]
        im = ax.imshow(model_ratio[..., 0], cmap='plasma', vmin=0, vmax=v_max)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Model')

        [ax.set_xticks([]) for ax in axs.flatten()]
        [ax.set_yticks([]) for ax in axs.flatten()]

        axs[0, 0].set_ylabel('tB')
        axs[1, 0].set_ylabel('pB')
        axs[2, 0].set_ylabel('Ratio')

        fig.tight_layout()
        wandb.log({f'images.{self.ds_key}': fig})
        plt.close('all')

        # self.plot_integrated_quantities(outputs['density'], outputs['distance'],
        #                                 outputs['z_vals_stratified'], outputs['z_vals_hierarchical'], outputs['distance_from_sun'],
        #                                 outputs['distance_from_obs'])

        val_loss = np.nanmean((model_image - target_image) ** 2)
        val_ssim = []
        for i in range(target_image.shape[-1]):
            val_ssim += [structural_similarity(np.nan_to_num(target_image[..., i], nan=0),
                                               np.nan_to_num(model_image[..., i], nan=0),
                                               data_range=1)]
        val_ssim = np.mean(val_ssim)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({f'validation.loss.{self.ds_key}': val_loss,
                   f'validation.ssim.{self.ds_key}': val_ssim,
                   f'validation.psnr.{self.ds_key}': val_psnr})

    def plot_integrated_quantities(self, density, distance, z_vals_stratified,
                                   z_vals_hierach, distance_from_sun, distance_from_obs):
        fig, axs = plt.subplots(1, 5, figsize=(24, 4))

        ax = axs[0]
        im = ax.imshow(density, cmap='inferno', norm='log')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'mean log(T)')

        ax = axs[1]
        im = ax.imshow(distance_from_sun, cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Total $n_e$')

        ax = axs[2]
        im = ax.imshow(distance_from_obs, cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Mean Absorption')

        # select index
        y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4  # select point in first quadrant
        plot_ray_sampling(z_vals_stratified[y, x] - distance, z_vals_hierach[y, x] - distance, axs[-1])

        fig.tight_layout()
        wandb.log({f'integrated_quantities.{self.ds_key}': fig})
        plt.close('all')


@rank_zero_only
def log_overview(images, poses, times, cmap, seconds_per_dt, Rs_per_ds, ref_date, ds_key=None):
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1] * Rs_per_ds
    colors = plt.get_cmap('viridis')(Normalize()(times))
    # fix arrow heads (2) + shaft color (2) --> 3 color elements
    cs = colors.tolist()
    for c in colors:
        cs.append(c)
        cs.append(c)

    norm = ImageNormalize(vmin=0, stretch=AsinhStretch(0.005), clip=True)

    iter_list = list(enumerate(images))
    step = max(1, len(iter_list) // 10)
    for i, img in iter_list[::step]:
        fig = plt.figure(figsize=(16, 8), dpi=100)
        ax = plt.subplot(121, projection='3d')
        # plot all viewpoints
        _ = ax.quiver(
            origins[..., 0].flatten(),
            origins[..., 1].flatten(),
            origins[..., 2].flatten(),
            dirs[..., 0].flatten(),
            dirs[..., 1].flatten(),
            dirs[..., 2].flatten(), color=cs, length=50, normalize=False, pivot='middle',
            linewidth=2, arrow_length_ratio=0.1, alpha=0.8)

        # plot current viewpoint
        _ = ax.quiver(
            origins[i:i + 1, ..., 0].flatten(),
            origins[i:i + 1, ..., 1].flatten(),
            origins[i:i + 1, ..., 2].flatten(),
            dirs[i:i + 1, ..., 0].flatten(),
            dirs[i:i + 1, ..., 1].flatten(),
            dirs[i:i + 1, ..., 2].flatten(), length=50, normalize=False, color='red', pivot='middle', linewidth=5,
            arrow_length_ratio=0.2)

        d = (1.2 * u.AU).to(u.solRad).value
        ax.set_xlim(-d, d)
        ax.set_ylim(-d, d)
        ax.set_zlim(-d, d)
        ax.scatter(0, 0, 0, marker='o', color='yellow')

        ax = plt.subplot(122)
        # plot corresponding image
        cmap = copy.deepcopy(get_cmap(cmap))
        cmap.set_bad('green', 1.)
        masked_img = np.ma.array(img[..., 0], mask=np.isnan(img[..., 0]))
        ax.imshow(masked_img, norm=norm, cmap=cmap, origin='lower')
        ax.set_axis_off()
        ax.set_title('Time: %s' % unnormalize_datetime(times[i], seconds_per_dt, ref_date).isoformat(' '))

        wandb.log({f'Overview.{ds_key}': fig})
        plt.close(fig)


def plot_ray_sampling(
        z_vals: torch.Tensor,
        z_hierarch: Optional[torch.Tensor] = None,
        ax: Optional[np.ndarray] = None):
    r"""
    Plot stratified and (optional) hierarchical samples.
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o', markersize=4)
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o', markersize=4)
    ax.set_ylim([-1, 2])
    # ax.set_xlim([-1.3, 1.3])
    ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)


class CubeCallback(BaseCallback):

    def __init__(self, cube_shape, Rs_per_ds, seconds_per_dt, **kwargs):
        super().__init__(**kwargs)
        self.cube_shape = cube_shape

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        rho_true = outputs['rho_true'].reshape(self.cube_shape).cpu().numpy()
        rho_pred = outputs['rho_pred'].reshape(self.cube_shape).cpu().numpy()

        flat_rho_true = rho_true.flatten()
        flat_rho_pred = rho_pred.flatten()

        corr_coeff = np.corrcoef(flat_rho_true, flat_rho_pred)[0, 1]
        wandb.log({f'valid.corr_coeff': corr_coeff})

        model = LinearRegression(fit_intercept=False)
        model.fit(flat_rho_pred.reshape(-1, 1), flat_rho_true)
        calibration_str = f'GT: {flat_rho_true.mean():.2E} SuNeRF: {flat_rho_pred.mean():.2E}; Coeff: {model.coef_[0]:.2E}'
        flat_rho_pred = model.predict(flat_rho_pred.reshape(-1, 1))

        mae = np.mean(np.abs(flat_rho_true - flat_rho_pred))
        wandb.log({f'valid.mae': mae})

        # 2D Histogram of the true and predicted densities

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.hist2d(np.log10(flat_rho_true), np.log10(flat_rho_pred), bins=100, cmap='viridis', norm=LogNorm())
        ax.set_xlabel('True Density [log N$_e$ cm$^{-3}$]')
        ax.set_ylabel('Predicted Density [log N$_e$ cm$^{-3}$]')
        ax.set_title(calibration_str)
        min_true, min_pred = np.min(np.log10(rho_true)), np.min(np.log10(rho_pred))
        max_true, max_pred = np.max(np.log10(rho_true)), np.max(np.log10(rho_pred))
        # plot 1:1 line
        ax.plot([min_true, max_true], [min_true, max_true], 'r--')

        fig.tight_layout()
        wandb.log({f'2D Histogram - {self.name}': wandb.Image(fig)})


class LatitudeSliceCallback(BaseCallback):

    def __init__(self, cube_shape, latitude, rho_normalization, Rs_per_ds, seconds_per_dt, **kwargs):
        super().__init__(**kwargs)
        self.latitude = np.deg2rad(latitude)
        self.cube_shape = cube_shape
        self.rho_normalization = 1.12e6  # TODO: rho_normalization
        self.velocity_normalization = (Rs_per_ds / seconds_per_dt) * (1 * u.solRad / u.s).to_value(u.km / u.s)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        spherical_coords = outputs['spherical_coords'].reshape(self.cube_shape + (3,)).cpu().numpy()
        rho_true = outputs['rho_true'].reshape(self.cube_shape).cpu().numpy()
        rho_pred = outputs['rho_pred'].reshape(self.cube_shape).cpu().numpy() * self.rho_normalization
        velocity = outputs['v_pred'].reshape(self.cube_shape + (3,)).cpu().numpy() * self.velocity_normalization

        lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 1] - self.latitude))

        r = spherical_coords[:, lat_idx, :, 0]
        ph = spherical_coords[:, lat_idx, :, 2]

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 5))

        ax = axs[0]
        z = rho_true[:, lat_idx, :]
        pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm='log', cmap='inferno')  # , vmin=1e1, vmax=1e3)
        fig.colorbar(pc, ax=ax, label='Density [N$_e$ cm$^{-3}$]')
        ax.set_title("Ground-truth", va='bottom')

        ax = axs[1]
        z = rho_pred[:, lat_idx, :]
        pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm='log', cmap='inferno')  # , vmin=1e1, vmax=1e3)
        fig.colorbar(pc, ax=ax, label='Density [N$_e$ cm$^{-3}$]')
        ax.set_title("SuNeRF", va='bottom')

        fig.tight_layout()
        wandb.log({f"Latitude={np.rad2deg(self.latitude).astype(int):03d} deg - Slice": wandb.Image(fig)})
        plt.close('all')


class VelocitySliceCallback(BaseCallback):

    def __init__(self, cube_shape, latitude, rho_normalization, Rs_per_ds, seconds_per_dt, plot_velocities=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.latitude = np.deg2rad(latitude)
        self.cube_shape = cube_shape
        self.rho_normalization = 1.12e6  # TODO: rho_normalization
        self.velocity_normalization = (Rs_per_ds / seconds_per_dt) * (1 * u.solRad / u.s).to_value(u.km / u.s)
        self.Rs_per_ds = Rs_per_ds
        self.plot_velocities = plot_velocities

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        query_points = outputs['query_points'].reshape(self.cube_shape + (4,)).cpu().numpy()
        query_points[..., :3] = query_points[..., :3] * self.Rs_per_ds
        rho_pred = outputs['rho_pred'].reshape(self.cube_shape).cpu().numpy() * self.rho_normalization
        velocity = outputs['v_pred'].reshape(self.cube_shape + (3,)).cpu().numpy() * self.velocity_normalization

        n_times = query_points.shape[3]
        max_radius = np.nanmax(query_points[..., :3])

        v_norm = np.linalg.norm(velocity, axis=-1)
        v_min, v_max = np.nanmin(v_norm), np.nanmax(v_norm)

        fig, axs = plt.subplots(2, n_times, figsize=(3 * n_times, 5), dpi=300)

        for i in range(n_times):
            ax = axs[0, i]
            im_rho = ax.imshow(rho_pred[:, :, 0, i], norm='log',
                               extent=[-max_radius, max_radius, -max_radius, max_radius], cmap='inferno',
                               origin='lower')

            if self.plot_velocities:
                # overlay velocity vectors
                quiver_pos = query_points[::8, ::8, 0, i,
                :2]  # block_reduce(query_points_npy, (8, 8, 1, 1, 1), np.mean)
                quiver_vel = velocity[::8, ::8, 0, i]  # block_reduce(velocity, (8, 8, 1), np.mean)
                ax.quiver(quiver_pos[:, :, 0], quiver_pos[:, :, 1],
                          quiver_vel[:, :, 0], quiver_vel[:, :, 1],
                          scale=10000,
                          color='white')

            ax = axs[1, i]
            im_v = ax.imshow(np.linalg.norm(velocity[:, :, 0, i], axis=-1), cmap='cividis',
                             extent=[-max_radius, max_radius, -max_radius, max_radius], origin='lower',
                             vmin=v_min, vmax=v_max)

        divider = make_axes_locatable(axs[0, -1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im_rho, cax=cax, label='N$_e$ / cm$^3$')

        divider = make_axes_locatable(axs[1, -1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im_v, cax=cax, label='km/s')

        fig.tight_layout()
        wandb.log({f"Velocity Slice - {self.name}": fig})
        plt.close('all')


class LongitudeSliceCallback(BaseCallback):

    def __init__(self, cube_shape, longitude, rho_normalization, Rs_per_ds, seconds_per_dt, **kwargs):
        super().__init__(**kwargs)
        self.longitude = np.deg2rad(longitude)
        self.cube_shape = cube_shape
        self.rho_normalization = 1.12e6  # TODO: rho_normalization
        self.velocity_normalization = (Rs_per_ds / seconds_per_dt) * (1 * u.solRad / u.s).to_value(u.km / u.s)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        spherical_coords = outputs['spherical_coords'].reshape(self.cube_shape + (3,)).cpu().numpy()
        rho_true = outputs['rho_true'].reshape(self.cube_shape).cpu().numpy()
        rho_pred = outputs['rho_pred'].reshape(self.cube_shape).cpu().numpy() * self.rho_normalization
        velocity = outputs['v_pred'].reshape(self.cube_shape + (3,)).cpu().numpy() * self.velocity_normalization

        lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 2] - self.longitude))

        r = spherical_coords[:, :, lon_idx, 0]
        th = spherical_coords[:, :, lon_idx, 1]

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 5))

        ax = axs[0]
        z = rho_true[:, :, lon_idx]
        pc = ax.pcolormesh(th, r, z, edgecolors='face', norm='log', cmap='inferno')  # , vmin=1e1, vmax=1e3)
        fig.colorbar(pc, ax=ax, label='Density [N$_e$ cm$^{-3}$]')
        ax.set_title("Ground-truth", va='bottom')

        ax = axs[1]
        z = rho_pred[:, :, lon_idx]
        pc = ax.pcolormesh(th, r, z, edgecolors='face', norm='log', cmap='inferno')  # , vmin=1e1, vmax=1e3)
        fig.colorbar(pc, ax=ax, label='Density [N$_e$ cm$^{-3}$]')
        ax.set_title("SuNeRF", va='bottom')

        fig.tight_layout()
        wandb.log(
            {f"{self.name} - Longitude={np.rad2deg(self.longitude).astype(int):03d} deg - Slice": wandb.Image(fig)})
        plt.close('all')


class ConditionedImageCallback(BaseCallback):

    def __init__(self, ds_key, image_shape, cmap='gray'):
        super().__init__(ds_key)
        self.image_shape = image_shape
        self.cmap = plt.get_cmap(cmap)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        predicted_image = outputs['predicted_image']
        target_image = outputs['target_image']

        self.plot_samples(predicted_image, outputs['height_map'], outputs['absorption_map'],
                          target_image, outputs['z_vals'], outputs['distance'].mean(), self.cmap)

        # handle NaNs
        predicted_image = np.nan_to_num(predicted_image, nan=0.0)
        target_image = np.nan_to_num(target_image, nan=0.0)

        val_loss = ((predicted_image - target_image) ** 2).mean()
        val_ssim = structural_similarity(target_image[..., 0], predicted_image[..., 0], data_range=1)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({'validation.loss': val_loss, 'validation.ssim': val_ssim, 'validation.psnr': val_psnr})

    def plot_samples(self, predicted_image, height_map, absorption_map, target_image, z_vals, distance, cmap):
        # Log example images on wandb
        # # Plot example outputs

        fig, ax = plt.subplots(1, 5, figsize=(30, 4))

        im = ax[0].imshow(target_image[..., 0], cmap=cmap)#, norm=sdo_img_norm)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title(f'Target')
        im = ax[1].imshow(predicted_image[..., 0], cmap=cmap)#, norm=sdo_img_norm)
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title(f'Predicted')
        im = ax[2].imshow(height_map, cmap='plasma')#, vmin=1, vmax=1.3)
        plt.colorbar(im, ax=ax[2])
        ax[2].set_title(f'Emission Height')
        im = ax[3].imshow(absorption_map, cmap='viridis')#, vmin=0)
        plt.colorbar(im, ax=ax[3])
        ax[3].set_title(f'Absorption')

        # select index
        y, x = z_vals.shape[0] // 4, z_vals.shape[1] // 4  # select point in first quadrant
        plot_ray_sampling(z_vals[y, x], None, ax[-1])

        wandb.log({"Comparison": fig})
        plt.close('all')
