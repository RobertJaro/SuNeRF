import copy
from typing import Optional

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import Callback
from skimage.metrics import structural_similarity

from sunerf.data.date_util import unnormalize_datetime
from sunerf.data.utils import sdo_img_norm


class BaseCallback(Callback):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_validation_outputs(self, pl_module):
        if self.name not in pl_module.validation_outputs:
            return None
        outputs = pl_module.validation_outputs[self.name]
        return outputs


class AbsorptionCallback(BaseCallback):

    def __init__(self, name, image_shape):
        super().__init__(name)
        self.image_shape = image_shape

    def on_validation_epoch_end(self, trainer, pl_module):
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

    def __init__(self, name, image_shape, cmap='gray'):
        super().__init__(name)
        self.image_shape = image_shape
        self.cmap = plt.get_cmap(cmap)
        self.normalize = ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005), clip=True)

    def on_validation_epoch_end(self, trainer, pl_module):
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
        plot_ray_sampling(z_vals_stratified[y, x] - distance, z_vals_hierach[y, x] - distance, ax[-1])

        wandb.log({"Comparison": fig})
        plt.close('all')


class PlasmaImageCallback(BaseCallback):

    def __init__(self, name, image_shape, cmaps=None):
        super().__init__(name)
        self.image_shape = image_shape
        self.cmaps = cmaps
        self.normalize = ImageNormalize(vmin=0, vmax=1, clip=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        # reshape
        outputs = {k: v.view(*self.image_shape, *v.shape[1:]).cpu().numpy() for k, v in outputs.items()}

        fine_image = outputs['fine_image']
        target_image = outputs['target_image']
        coarse_image = outputs['coarse_image']

        cmaps = self.cmaps
        cmaps = ['gray'] * fine_image.shape[0] if cmaps is None else cmaps
        for i, cmap in enumerate(cmaps):
            cmap = plt.get_cmap(cmap)
            self.plot_samples(fine_image[..., i], coarse_image[..., i],
                              target_image[..., i], outputs['z_vals_stratified'], outputs['z_vals_hierarchical'],
                              outputs['distance'].mean(), cmap, title=f'{cmap.name}')
        self.plot_integrated_quantities(outputs['height_map'], outputs['mean_T'], outputs['total_ne'],
                                        outputs['mean_absorption'], )

        val_loss = ((fine_image - target_image) ** 2).mean()
        val_ssim = []
        for i in range(target_image.shape[-1]):
            val_ssim += [structural_similarity(target_image[..., i], fine_image[..., i], data_range=1)]
        val_ssim = np.mean(val_ssim)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({'validation.loss': val_loss, 'validation.ssim': val_ssim, 'validation.psnr': val_psnr})

    def plot_integrated_quantities(self, height_map, mean_T, total_ne, absorption):
        fig, axs = plt.subplots(1, 4, figsize=(20, 4))

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

        fig.tight_layout()
        wandb.log({'integrated_quantities': fig})
        plt.close('all')

    def plot_samples(self, fine_image, coarse_image, target_image, z_vals_stratified,
                     z_vals_hierach, distance, cmap, title='comparison'):
        # Log example images on wandb
        # # Plot example outputs

        fig, ax = plt.subplots(1, 4, figsize=(20, 4))

        v_max = np.nanmax(target_image)
        im = ax[0].imshow(target_image, cmap=cmap, vmin=0, vmax=v_max)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[0].set_title(f'Target')

        im = ax[1].imshow(fine_image, cmap=cmap, vmin=0, vmax=v_max)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[1].set_title(f'Fine')

        im = ax[2].imshow(coarse_image, cmap=cmap, vmin=0, vmax=v_max)
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[2].set_title(f'Coarse')

        # select index
        y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4  # select point in first quadrant
        plot_ray_sampling(z_vals_stratified[y, x] - distance, z_vals_hierach[y, x] - distance, ax[-1])

        fig.tight_layout()
        wandb.log({f"{title}": fig})
        plt.close('all')


def log_overview(images, poses, times, cmap, seconds_per_dt, ref_time):
    dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
    origins = poses[:, :3, -1]
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
            linewidth=2, arrow_length_ratio=0.1)

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
        ax.set_title('Time: %s' % unnormalize_datetime(times[i], seconds_per_dt, ref_time).isoformat(' '))

        wandb.log({'Overview': fig})
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
