import copy
from typing import Optional

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import Callback
from skimage.metrics import structural_similarity
from sklearn.linear_model import LinearRegression

from sunerf.data.date_util import unnormalize_datetime
from sunerf.data.utils import sdo_img_norm
from pytorch_lightning.utilities import rank_zero_only

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
        wandb.log({'absorption': wandb.Image(fig)})
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

        prediction_image = self.normalize(outputs['fine_image'])
        target_image = self.normalize(outputs['target_image'])
        coarse_image = self.normalize(outputs['coarse_image'])

        self.plot_samples(prediction_image, coarse_image, outputs['height_map'], outputs['absorption_map'],
                          target_image, outputs['z_vals_stratified'], outputs['z_vals_hierarchical'],
                          outputs['distance'].mean(), self.cmap)

        val_loss = ((prediction_image - target_image) ** 2).mean()
        val_ssim = structural_similarity(target_image[..., 0], prediction_image[..., 0], data_range=1)
        val_psnr = -10. * np.log10(val_loss)

        wandb.log({'validation.loss': val_loss, 'validation.ssim': val_ssim, 'validation.psnr': val_psnr})

    def plot_samples(self, prediction_image, coarse_image, height_map, absorption_map, target_image, z_vals_stratified,
                     z_vals_hierach, distance, cmap):
        # Log example images on wandb
        # # Plot example outputs

        fig, ax = plt.subplots(1, 6, figsize=(30, 4))

        ax[0].imshow(target_image[..., 0], cmap=cmap, norm=sdo_img_norm)
        ax[0].set_title(f'Ground Truth')
        ax[1].imshow(prediction_image[..., 0], cmap=cmap, norm=sdo_img_norm)
        ax[1].set_title(f'Prediction')
        ax[2].imshow(coarse_image[..., 0], cmap=cmap, norm=sdo_img_norm)
        ax[2].set_title(f'Coarse')
        ax[3].imshow(height_map, cmap='plasma', vmin=1, vmax=1.3)
        ax[3].set_title(f'Emission Height')
        ax[4].imshow(absorption_map, cmap='viridis', vmin=0)
        ax[4].set_title(f'Absorption')

        # select index
        y, x = z_vals_stratified.shape[0] // 4, z_vals_stratified.shape[1] // 4  # select point in first quadrant
        plot_ray_sampling(z_vals_stratified[y, x], z_vals_hierach[y, x], ax[-1])

        wandb.log({"Comparison": wandb.Image(fig)})
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
            col[0].set_title(f'Ground Truth')

            im = col[1].imshow(pred_image[..., i], cmap=cmap, vmin=0, vmax=v_max)
            divider = make_axes_locatable(col[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            col[1].set_title(f'Prediction')

        [ax.set_axis_off() for ax in axs.flatten()]

        fig.tight_layout()
        wandb.log({f'images.{self.ds_key}': wandb.Image(fig)})
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
        wandb.log({f'integrated_quantities.{self.ds_key}': wandb.Image(fig)})
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

        fig, axs = plt.subplots(3, 2, figsize=(7, 9))

        # pB and tB images
        for i in range(2):
            ax = axs[i, 0]

            if np.isnan(target_image[..., i]).all():
                v_min = np.nanmin(model_image[..., i])
                v_max = np.nanmax(model_image[..., i])
            else:
                v_min = np.nanmin(target_image[..., i])
                v_max = np.nanmax(target_image[..., i])
            im = ax.imshow(target_image[..., i], cmap='plasma', vmin=v_min, vmax=v_max, origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title(f'Ground Truth')

            ax = axs[i, 1]
            im = ax.imshow(model_image[..., i], cmap='plasma', vmin=v_min, vmax=v_max, origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title(f'Prediction')

        # ratio images
        ax = axs[2, 0]
        if np.isnan(target_ratio[..., 0]).all():
            v_max = np.nanmax(model_ratio[..., 0])
        else:
            v_max = np.nanmax(target_ratio[..., 0])
        im = ax.imshow(target_ratio[..., 0], cmap='plasma', vmin=0, vmax=v_max, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Ground Truth')

        ax = axs[2, 1]
        im = ax.imshow(model_ratio[..., 0], cmap='plasma', vmin=0, vmax=v_max, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'Prediction')

        [ax.set_xticks([]) for ax in axs.flatten()]
        [ax.set_yticks([]) for ax in axs.flatten()]

        axs[0, 0].set_ylabel('tB')
        axs[1, 0].set_ylabel('pB')
        axs[2, 0].set_ylabel('Ratio')

        fig.tight_layout()
        wandb.log({f'images.{self.ds_key}': wandb.Image(fig)})
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


class CorrectionImageCallback(BaseCallback):

    def __init__(self, ds_key, image_shape):
        super().__init__(ds_key)
        self.image_shape = image_shape

        self.correction_keys = [
            "correction.f_corona",
            "correction.transmission",
            "correction.tB_add",
            "correction.pB_add",
            "correction.tB_straylight",
            "correction.pB_straylight",
            "correction.tB_mul",
            "correction.pB_mul",
            "correction.img",
            "correction.calibration_gain",
            "correction.calibration_offset",
            "correction.leakage",
        ]

        self.title_map = {
            "correction.f_corona": "F-Corona",
            "correction.transmission": "Transmission",
            "correction.tB_add": "tB Additive (Signed)",
            "correction.pB_add": "pB Additive (Signed)",
            "correction.tB_straylight": "tB Straylight (+)",
            "correction.pB_straylight": "pB Straylight (+)",
            "correction.tB_mul": "tB Multiplicative",
            "correction.pB_mul": "pB Multiplicative",
            "correction.img": "Input Image",
            "correction.calibration_gain": "Calibration Gain",
            "correction.calibration_offset": "Calibration Offset",
            "correction.leakage": "Leakage",
        }

        self.signed_additive_fields = {
            "correction.tB_add",
            "correction.pB_add",
            "correction.calibration_offset",
        }

        self.positive_additive_fields = {
            "correction.tB_straylight",
            "correction.pB_straylight",
        }

        self.multiplicative_fields = {
            "correction.tB_mul",
            "correction.pB_mul",
            "correction.calibration_gain",
            "correction.transmission",
        }

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None:
            return

        outputs = {
            k: v.view(*self.image_shape, *v.shape[1:]).detach().cpu().numpy()
            for k, v in outputs.items()
        }

        keys_present = [k for k in self.correction_keys if k in outputs]
        if not keys_present:
            return

        n = len(keys_present)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 8), squeeze=False)
        axes = axes[0]

        for ax, k in zip(axes, keys_present):

            img = outputs[k]
            img2d = img[..., 0] if (img.ndim >= 3 and img.shape[-1] >= 1) else img
            img2d = np.asarray(img2d)

            vmin = np.nanmin(img2d)
            vmax = np.nanmax(img2d)

            # ------------------------------------------------
            # 1) Signed additive corrections -> centered at 0
            # ------------------------------------------------
            if k in self.signed_additive_fields:
                vmax_abs = np.nanmax(np.abs(img2d))
                if not np.isfinite(vmax_abs) or vmax_abs <= 0.0:
                    vmax_abs = 1e-8
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax_abs, vmax=vmax_abs)
                cmap = "RdBu_r"

            # ------------------------------------------------
            # 2) Positive additive corrections -> positive, log scale
            # ------------------------------------------------
            elif k in self.positive_additive_fields:
                vmin_plot = max(np.nanmin(img2d), 1e-12)
                vmax_plot = max(np.nanmax(img2d), vmin_plot * (1.0 + 1e-6))
                norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot)
                cmap = "Reds"

            # ------------------------------------------------
            # 3) Multiplicative corrections -> centered at 1
            # ------------------------------------------------
            elif k in self.multiplicative_fields:
                deviation = img2d - 1.0
                vmax_abs = np.nanmax(np.abs(deviation))
                if not np.isfinite(vmax_abs) or vmax_abs <= 0.0:
                    vmax_abs = 1e-8
                norm = TwoSlopeNorm(vcenter=1.0,
                                    vmin=1.0 - vmax_abs,
                                    vmax=1.0 + vmax_abs)
                cmap = "RdBu_r"

            # ------------------------------------------------
            # 4) F-corona -> positive, log scale
            # ------------------------------------------------
            elif k == "correction.f_corona":
                vmin_plot = max(np.nanmin(img2d), 1e-12)
                vmax_plot = max(np.nanmax(img2d), vmin_plot * (1.0 + 1e-6))
                norm = LogNorm(vmin=vmin_plot, vmax=vmax_plot)
                cmap = "Reds"

            # ------------------------------------------------
            # 5) Leakage -> bounded [0,1], Reds
            # ------------------------------------------------
            elif k == "correction.leakage":
                norm = Normalize(vmin=0.0, vmax=0.1)
                cmap = "Reds"

            # ------------------------------------------------
            # 6) Fallback
            # ------------------------------------------------
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = "viridis"

            im = ax.imshow(img2d, cmap=cmap, norm=norm, origin='lower')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            ax.set_title(self.title_map.get(k, k))
            ax.set_axis_off()

        fig.tight_layout()
        wandb.log({f"correction.{self.ds_key}": wandb.Image(fig)})
        plt.close(fig)


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

    def _channel_lognorm(data):
        positive = data[np.isfinite(data) & (data > 0)]
        if positive.size == 0:
            return None
        vmin = float(np.nanmin(positive))
        vmax = float(np.nanmax(positive))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if vmax <= vmin:
            vmax = np.nextafter(vmin, np.inf)
        return LogNorm(vmin=vmin, vmax=vmax)

    tb_norm = _channel_lognorm(images[..., 0] if images.ndim == 4 else images)
    pb_norm = _channel_lognorm(images[..., 1]) if images.ndim == 4 and images.shape[-1] > 1 else None

    def _imshow_log(ax, data2d, title, norm):
        finite = np.isfinite(data2d)
        good = finite & (data2d > 0)
        if not np.any(finite):
            ax.set_axis_off()
            return None
        if not np.any(good) or norm is None:
            ax.set_axis_off()
            return None
        cm = copy.deepcopy(get_cmap(cmap))
        cm.set_bad('green', 1.)
        masked = np.ma.array(data2d, mask=~good)
        im = ax.imshow(masked, norm=norm, cmap=cm, origin='lower')
        ax.set_axis_off()
        ax.set_title(title)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        return im

    iter_list = list(enumerate(images))
    step = max(1, len(iter_list) // 10)
    for i, img in iter_list[::step]:
        # detect availability of pB
        has_pb = (img.ndim >= 3) and (img.shape[-1] > 1)

        fig = plt.figure(figsize=((16, 8) if not has_pb else (22, 8)), dpi=100)

        # --- left: 3D overview (unchanged) ---
        ax = plt.subplot(1, 2 + int(has_pb), 1, projection='3d')

        _ = ax.quiver(
            origins[..., 0].flatten(),
            origins[..., 1].flatten(),
            origins[..., 2].flatten(),
            dirs[..., 0].flatten(),
            dirs[..., 1].flatten(),
            dirs[..., 2].flatten(),
            color=cs, length=50, normalize=False, pivot='middle',
            linewidth=2, arrow_length_ratio=0.1, alpha=0.8)

        _ = ax.quiver(
            origins[i:i + 1, ..., 0].flatten(),
            origins[i:i + 1, ..., 1].flatten(),
            origins[i:i + 1, ..., 2].flatten(),
            dirs[i:i + 1, ..., 0].flatten(),
            dirs[i:i + 1, ..., 1].flatten(),
            dirs[i:i + 1, ..., 2].flatten(),
            length=50, normalize=False, color='red', pivot='middle',
            linewidth=5, arrow_length_ratio=0.2)

        d = (1.2 * u.AU).to(u.solRad).value
        ax.set_xlim(-d, d)
        ax.set_ylim(-d, d)
        ax.set_zlim(-d, d)
        ax.scatter(0, 0, 0, marker='o', color='yellow')

        tstr = unnormalize_datetime(times[i], seconds_per_dt, ref_date).isoformat(' ')
        obs = origins[i]
        obs_r = np.linalg.norm(obs) + 1e-8
        obs_lat = np.rad2deg(np.arcsin(np.clip(obs[2] / obs_r, -1.0, 1.0)))
        obs_lon = (np.rad2deg(np.arctan2(obs[1], obs[0])) + 360.0) % 360.0
        fig.suptitle(f"Observer lon={obs_lon:.2f} deg, lat={obs_lat:.2f} deg | Time: {tstr}", fontsize=11)

        # --- right: images ---
        ax = plt.subplot(1, 2 + int(has_pb), 2)
        _imshow_log(ax, img[..., 0], f"tB | Time: {tstr}", tb_norm)

        if has_pb:
            ax = plt.subplot(1, 2 + int(has_pb), 3)
            _imshow_log(ax, img[..., 1], f"pB | Time: {tstr}", pb_norm)

        wandb.log({f'Overview.{ds_key}': wandb.Image(fig)})
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
    ax.set_title('Ray Samples: Stratified (blue), Hierarchical (red)')
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
        wandb.log({f"Velocity Slice - {self.name}": wandb.Image(fig)})
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


class RadialSlicesCallback(BaseCallback):
    """
    time × radius panels (latitude vs longitude),
    plus a dedicated last column for colorbars (one per time row).

    Layout:
      - Nt rows (time)
      - Nr columns (r slices) + 1 extra column (colorbars)
      - one colorbar per row in the last column
      - uses constrained_layout
    """

    def __init__(self, cube_shape, radii, rho_normalization, **kwargs):
        """
        Parameters
        ----------
        cube_shape : tuple
            (Nr, Nlat, Nlon, Nt)
        radii : array-like
            Radii in R_sun corresponding to the Nr slices.
        """
        super().__init__(**kwargs)
        self.cube_shape = cube_shape
        self.radii = np.asarray(radii, dtype=np.float32)
        self.rho_normalization = float(rho_normalization)

        if len(self.radii) != cube_shape[0]:
            raise ValueError(
                f"Number of radii ({len(self.radii)}) does not match Nr ({cube_shape[0]})."
            )

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        out = self.get_validation_outputs(pl_module)
        if out is None:
            return

        Nr, Nlat, Nlon, Nt = self.cube_shape

        rho = out["rho_pred"].detach().cpu().numpy().reshape(-1) * self.rho_normalization
        rho = rho.reshape(Nr, Nlat, Nlon, Nt)

        sph = out["spherical_coords"].detach().cpu().numpy().reshape(-1, 3)
        sph = sph.reshape(Nr, Nlat, Nlon, Nt, 3)

        lat = sph[0, :, 0, 0, 1]
        lon = sph[0, 0, :, 0, 2]

        extent = [
            np.rad2deg(lon.min()),
            np.rad2deg(lon.max()),
            np.rad2deg(lat.min()),
            np.rad2deg(lat.max()),
        ]

        norm = LogNorm(vmin=np.nanmin(rho), vmax=np.nanmax(rho))

        fig = plt.figure(
            figsize=(3.2 * (Nr + 1), 2.6 * Nt),
            constrained_layout=True,
            dpi=180,
        )

        # mosaic with dedicated colorbar column
        layout = []
        for it in range(Nt):
            row = [f"rho_t{it}_r{ir}" for ir in range(Nr)] + [f"cbar_t{it}"]
            layout.append(row)

        per_subplot_kw = {
            **{f"rho_t{it}_r{ir}": {} for it in range(Nt) for ir in range(Nr)},
            **{f"cbar_t{it}": {} for it in range(Nt)},
        }

        axd = fig.subplot_mosaic(
            layout,
            per_subplot_kw=per_subplot_kw,
            width_ratios=[1.0] * Nr + [0.05],
        )

        for it in range(Nt):
            mappable_row = None

            for ir in range(Nr):
                ax = axd[f"rho_t{it}_r{ir}"]

                img = rho[ir, :, :, it]

                mappable_row = ax.imshow(
                    img,
                    origin="lower",
                    extent=extent,
                    norm=norm,
                    cmap="inferno",
                )

                if it == 0:
                    ax.set_title(f"{self.radii[ir]:.2f} R$_\\odot$")

                if ir == 0:
                    ax.set_ylabel(f"t#{it}")

                ax.set_xlabel("Longitude (deg)")
                ax.set_ylabel("Latitude (deg)")

            # dedicated colorbar axis
            cax = axd[f"cbar_t{it}"]

            fig.colorbar(
                mappable_row,
                cax=cax,
                orientation="vertical",
                label=r"Density [N$_e$ cm$^{-3}$]",
            )

        wandb.log({f"radial_slices.{self.name}": wandb.Image(fig)})
        plt.close(fig)


class LongitudeSlicesCallback(BaseCallback):
    """
    time × longitude polar density panels, plus a dedicated last column for colorbars.

    Layout:
      - Nt rows (time)
      - Nlon columns (longitude slices) + 1 extra column (colorbars)
      - one colorbar per row (i.e., per time) in the last column

    Notes:
      - Uses constrained_layout=True and allocates explicit "cbar" axes via subplot_mosaic.
      - Colorbars sit in the dedicated last column (all the way to the right).
    """

    def __init__(self, cube_shape, rho_normalization, Rs_per_ds, seconds_per_dt,
                 longitude_deg=(0, 30, 60, 90, 120, 150), **kwargs):
        super().__init__(**kwargs)
        self.cube_shape = cube_shape
        self.longitude_deg = np.asarray(longitude_deg, dtype=np.float32)
        self.rho_normalization = float(rho_normalization)
        self.velocity_normalization = float((Rs_per_ds * u.solRad / (seconds_per_dt * u.s)).to_value(u.km / u.s))

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        out = self.get_validation_outputs(pl_module)
        if out is None:
            return

        Nr, Nlat, Nlon, Nt = self.cube_shape

        rho = out["rho_pred"].detach().cpu().numpy().reshape(-1) * self.rho_normalization
        rho = rho.reshape(Nr, Nlat, Nlon, Nt)

        sph = out["spherical_coords"].detach().cpu().numpy().reshape(-1, 3)
        sph = sph.reshape(Nr, Nlat, Nlon, Nt, 3)

        longitudes_deg = (
            self.longitude_deg
            if len(self.longitude_deg) == Nlon
            else np.rad2deg(sph[0, 0, :, 0, 2])
        )

        density_norm = LogNorm(vmin=np.nanmin(rho), vmax=np.nanmax(rho))

        fig = plt.figure(
            figsize=(4.1 * (Nlon + 1), 3.5 * Nt),
            constrained_layout=True,
            dpi=180,
        )

        # --- mosaic: add last column for colorbars (one cbar axis per row) ---
        layout = []
        for it in range(Nt):
            row = [f"rho_t{it}_j{j}" for j in range(Nlon)] + [f"cbar_t{it}"]
            layout.append(row)

        per_subplot_kw = {
            **{f"rho_t{it}_j{j}": {"projection": "polar"} for it in range(Nt) for j in range(Nlon)},
            **{f"cbar_t{it}": {} for it in range(Nt)},
        }

        # make the cbar column skinny
        axd = fig.subplot_mosaic(
            layout,
            per_subplot_kw=per_subplot_kw,
            width_ratios=[1.0] * Nlon + [0.06],
        )

        for it in range(Nt):
            mappable_row = None

            for j in range(Nlon):
                ax = axd[f"rho_t{it}_j{j}"]

                img = rho[:, :, j, it]  # (Nr, Nlat)
                r = sph[:, :, j, it, 0]  # (Nr, Nlat)
                lat = sph[:, :, j, it, 1]  # (Nr, Nlat)

                mappable_row = ax.pcolormesh(
                    lat, r, img,
                    shading="auto",
                    norm=density_norm,
                    cmap="inferno",
                )

                if it == 0:
                    ax.set_title(f"{float(longitudes_deg[j]):.1f}°", pad=10)

                if j == 0:
                    ax.text(
                        -0.15, 0.5, f"t#{it}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                    )

                ax.set_xlabel("Latitude (rad)")
                ax.set_ylabel(r"Radius (R$_\odot$)")
                ax.set_rlim((0, None))

            # one colorbar per *row* in the dedicated last column
            cax = axd[f"cbar_t{it}"]

            fig.colorbar(
                mappable_row,
                cax=cax,
                orientation="vertical",
                label=r"Density [N$_e$ cm$^{-3}$]",
            )

        wandb.log({f"longitude_slices.rho.{self.name}": wandb.Image(fig)})
        plt.close(fig)

        if "v_pred" not in out:
            print('no velocity predictions found, skipping velocity plots')
            return

        v = out["v_pred"].detach().cpu().numpy().reshape(Nr, Nlat, Nlon, Nt, 3)
        vmag = np.linalg.norm(v, axis=-1) * self.velocity_normalization
        vmag = np.clip(vmag, 1e-30, None)
        vmag_norm = LogNorm(vmin=np.nanmin(vmag), vmax=np.nanmax(vmag))

        fig_v = plt.figure(
            figsize=(4.1 * (Nlon + 1), 3.5 * Nt),
            constrained_layout=True,
            dpi=180,
        )

        axd_v = fig_v.subplot_mosaic(
            layout,
            per_subplot_kw=per_subplot_kw,
            width_ratios=[1.0] * Nlon + [0.06],
        )

        for it in range(Nt):
            mappable_row_v = None

            for j in range(Nlon):
                ax = axd_v[f"rho_t{it}_j{j}"]

                img_v = vmag[:, :, j, it]
                r = sph[:, :, j, it, 0]
                lat = sph[:, :, j, it, 1]

                mappable_row_v = ax.pcolormesh(
                    lat, r, img_v,
                    shading="auto",
                    norm=vmag_norm,
                    cmap="cividis",
                )

                if it == 0:
                    ax.set_title(f"{float(longitudes_deg[j]):.1f}°", pad=10)

                if j == 0:
                    ax.text(
                        -0.15, 0.5, f"t#{it}",
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                    )

                ax.set_xlabel("Latitude (rad)")
                ax.set_ylabel(r"Radius (R$_\odot$)")
                ax.set_rlim((0, None))

            cax = axd_v[f"cbar_t{it}"]
            fig_v.colorbar(
                mappable_row_v,
                cax=cax,
                orientation="vertical",
                label=r"|v| [km s$^{-1}$]",
            )

        wandb.log({f"longitude_slices.vmag.{self.name}": wandb.Image(fig_v)})
        plt.close(fig_v)


class LongitudeTimeVelocityMagCallback(BaseCallback):
    """
    Same grid as density but plots |v| in km/s.
    Expects v_pred (...,3) in model units (ds/dt).
    """

    def __init__(self, cube_shape, Rs_per_ds, seconds_per_dt, **kwargs):
        super().__init__(**kwargs)
        self.cube_shape = cube_shape
        self.velocity_normalization = float((Rs_per_ds * u.solRad / (seconds_per_dt * u.s)).to_value(u.km / u.s))

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        out = self.get_validation_outputs(pl_module)
        if out is None:
            return

        Nlon, Nt, Nr, Nth = self.cube_shape
        v = out["v_pred"].view(Nlon, Nt, Nr, Nth, 3).cpu().numpy()
        vmag = np.linalg.norm(v, axis=-1) * self.velocity_normalization

        sph = out["spherical_coords"].view(Nlon, Nt, Nr, Nth, 3).cpu().numpy()
        r = sph[0, 0, :, 0, 0]
        th = sph[0, 0, 0, :, 1]
        th_deg = np.rad2deg(th)

        times = out.get("meta.times", None)
        times = times.cpu().numpy() if times is not None else None
        lon = out.get("meta.longitudes_rad", None)
        lon = np.rad2deg(lon.cpu().numpy()) if lon is not None else np.arange(Nlon)

        fig, axes = plt.subplots(Nlon, Nt, figsize=(4 * Nt, 3 * Nlon), squeeze=False)
        for i in range(Nlon):
            for j in range(Nt):
                ax = axes[i, j]
                img = vmag[i, j, :, :]
                ax.imshow(img, origin="lower", aspect="auto",
                          extent=[th_deg.min(), th_deg.max(), r.min(), r.max()])
                if i == 0:
                    ax.set_title(f"t={times[j]:.3f}" if times is not None else f"t#{j}")
                if j == 0:
                    ax.set_ylabel(f"r [R☉]\nlon={lon[i]:.0f}°")
                if i == Nlon - 1:
                    ax.set_xlabel("lat [deg]")

        fig.tight_layout()
        wandb.log({f"longitude_time_velocitymag.{self.name}": wandb.Image(fig)})
        plt.close(fig)


class FixedViewpointSeriesCallback(BaseCallback):
    """
    3 rows: tB, pB, density
    N cols: time snapshots
    Expects instrument validation outputs: model_image (...,2), density (...,1 or ...), target optional NaNs
    """

    def __init__(self, image_shape, n_times=6, **kwargs):
        super().__init__(**kwargs)
        self.image_shape = image_shape
        self.n_times = int(n_times)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        out = self.get_validation_outputs(pl_module)
        if out is None:
            return

        H, W = self.image_shape
        model_image = out["model_image"].view(self.n_times, H, W, -1).cpu().numpy()
        density = out["density"].view(self.n_times, H, W, -1).cpu().numpy()

        fig, axes = plt.subplots(3, self.n_times, figsize=(4 * self.n_times, 10), squeeze=False)

        for j in range(self.n_times):
            axes[0, j].imshow(model_image[j, :, :, 0], origin="lower")
            axes[0, j].set_title(f"tB t#{j}")
            axes[1, j].imshow(model_image[j, :, :, 1], origin="lower")
            axes[1, j].set_title(f"pB t#{j}")
            d = density[j, :, :, 0] if density.shape[-1] >= 1 else density[j, :, :, 0]
            axes[2, j].imshow(np.log10(np.clip(d, 1e-30, None)), origin="lower")
            axes[2, j].set_title(f"log10 rho t#{j}")

            for i in range(3):
                axes[i, j].axis("off")

        fig.tight_layout()
        wandb.log({f"fixed_viewpoint_series.{self.name}": wandb.Image(fig)})
        plt.close(fig)


class StarBackgroundCallback(BaseCallback):
    """
    Visualize StarBackgroundModule output (additive background) in log scale.
    Expects outputs['background'] shaped (Npix, 2) after validation_step.
    """

    def __init__(self, ds_key, image_shape, eps=1e-12):
        super().__init__(ds_key)
        self.image_shape = tuple(image_shape)
        self.eps = float(eps)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self.get_validation_outputs(pl_module)
        if outputs is None or 'background' not in outputs:
            return

        bg = outputs['background'].view(*self.image_shape, -1).detach().cpu().numpy()
        # bg[...,0]=tB additive, bg[...,1]=pB additive (if present)
        has_pb = (bg.shape[-1] > 1)

        def _imshow_log(ax, img, title):
            img = np.asarray(img)
            good = np.isfinite(img) & (img > 0)
            if not np.any(good):
                ax.set_axis_off()
                ax.set_title(title + " (no >0 finite)")
                return None
            masked = np.ma.array(img, mask=~good)
            im = ax.imshow(masked, origin='lower', cmap='magma',
                           norm=LogNorm(vmin=max(self.eps, masked.min()), vmax=masked.max()))
            ax.set_title(title)
            ax.set_axis_off()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            return im

        ncols = 2 if has_pb else 1
        fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

        if ncols == 1:
            axs = [axs]

        _imshow_log(axs[0], bg[..., 0], "Star background (tB)")
        if has_pb:
            _imshow_log(axs[1], bg[..., 1], "Star background (pB)")

        fig.tight_layout()
        wandb.log({f"star_background.{self.ds_key}": wandb.Image(fig)})
        plt.close(fig)


class FullStarBackgroundCallback(BaseCallback):
    """
    Plots star background image(s) as a full-sky latitude/longitude map in log scale.
    Expects outputs: background (...,C). Usually C=2 for (tB,pB).
    """

    def __init__(self, ds_key, image_shape, eps=1e-12, name=None):
        super().__init__(ds_key=ds_key, name=name)
        self.image_shape = image_shape
        self.eps = float(eps)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        out = self.get_validation_outputs(pl_module)
        if out is None or "background" not in out:
            return

        H, W = self.image_shape
        bg = out["background"].view(H, W, -1).detach().cpu().numpy()

        nC = bg.shape[-1]
        fig, axes = plt.subplots(1, nC, figsize=(6 * nC, 5), squeeze=False)
        axes = axes[0]
        extent = [0, 360, -90, 90]

        for c in range(nC):
            img = np.asarray(bg[..., c])
            img = np.clip(img, self.eps, None)
            axes[c].imshow(img, origin="lower", aspect="auto", extent=extent,
                           norm=LogNorm(vmin=np.nanpercentile(img, 5), vmax=np.nanpercentile(img, 99)))
            channel = "tB" if c == 0 else "pB" if c == 1 else f"ch{c}"
            axes[c].set_title(f"Star background {channel} (log)")
            axes[c].set_xlabel("Longitude [deg]")
            axes[c].set_ylabel("Latitude [deg]")

        fig.tight_layout()
        wandb.log({f"star_background_full.{self.name}": wandb.Image(fig)})
        plt.close(fig)
