import os

import torch
from astropy import units as u
from torch import nn

from sunerf.data.loader.base_loader import BaseDataModule
from sunerf.model.model import RhoModel
from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.util import jacobian
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.thomson import ThomsonScattering
from sunerf.train.correction import CorrectionModule, CalibrationModule, AlignmentModule, StarBackgroundModule
from sunerf.train.render_mode import RenderMode
from sunerf.train.scaling import ImageAsinhScaling, ImageLinearScaling, ImageLogScaling


class ThomsonSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt,
                 instruments, lambda_config=None,
                 sampling_config=None,
                 model_config=None,
                 shuffle_config=None, **kwargs):
        # setup rendering
        sampling_config = sampling_config if sampling_config is not None else {}

        rendering_modules = {}
        scaling_modules = {}
        correction_modules = {}
        calibration_modules = {}
        alignment_modules = {}
        background_modules = {}
        for instrument_config in instruments:
            instrument_config = instrument_config.copy()
            instrument_key = instrument_config.pop('key')
            instrument_type = instrument_config.pop('type')
            correction = instrument_config.pop('correction', False)
            calibration = instrument_config.pop('calibration', False)
            alignment = instrument_config.pop('alignment', False)
            background = instrument_config.pop('background', False)
            # rendering module
            rendering_config = instrument_config.pop('rendering', {})
            if instrument_type == 'default':
                rendering_modules[instrument_key] = ThomsonScattering(Rs_per_ds=Rs_per_ds, **rendering_config)
            else:
                raise ValueError(f"Unknown instrument type: {instrument_type}")
            # image scaling
            scaling_config = instrument_config.pop('scaling', {})
            scaling_type = scaling_config.pop('type', 'asinh')
            if scaling_type == 'asinh':
                scaling_modules[instrument_key] = ImageAsinhScaling(**scaling_config)
            elif scaling_type == 'linear':
                scaling_modules[instrument_key] = ImageLinearScaling(**scaling_config)
            elif scaling_type == 'log':
                scaling_modules[instrument_key] = ImageLogScaling(**scaling_config)
            else:
                raise ValueError(f"Unknown scaling type: {scaling_type}")
            # correction module
            if correction:
                correction_config = {} if isinstance(correction, bool) else correction
                correction_modules[instrument_key] = CorrectionModule(**correction_config)
            # calibration module
            if calibration:
                calibration_config = {} if isinstance(calibration, bool) else calibration
                calibration_modules[instrument_key] = CalibrationModule(**calibration_config)
            # alignment module
            if alignment:
                alignment_config = {} if isinstance(alignment, bool) else alignment
                alignment_modules[instrument_key] = AlignmentModule(**alignment_config)
            # star background
            if background:
                background_config = {} if isinstance(background, bool) else background
                background_modules[instrument_key] = StarBackgroundModule(**background_config)

        model_config = {} if model_config is None else model_config
        model = RhoModel(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, **model_config)
        rendering = BasicRenderingModule(model=model,
                                         rendering_modules=rendering_modules,
                                         Rs_per_ds=Rs_per_ds,
                                         sampling_config=sampling_config, shuffle_config=shuffle_config)

        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        self.rendering_modules = rendering_modules
        self.model = model

        # define lambda values
        lambda_config = {'image': 1.0,
                         'ratio': 1.0,
                         'continuity': 1e-3,
                         'radial': 1e-2,
                         'velocity': 1e-3} if lambda_config is None else lambda_config
        # check lambda config
        available_lambdas = ['image', 'ratio', 'continuity', 'radial', 'velocity',
                             'calibration',
                             'f_corona', 'transmission', 'calibration_gain', 'calibration_offset', 'calibration_scalar',
                             'pB_mul', 'tB_mul', 'pB_add', 'tB_add', 'pB_add_mean', 'tB_add_mean',
                             'pB_straylight', 'tB_straylight', 'star_background']
        for k in lambda_config:
            if k not in available_lambdas:
                raise ValueError(f"Unknown lambda_config key: {k}")
        # set lambdas default to 0.0
        for k in available_lambdas:
            if k not in lambda_config:
                lambda_config[k] = 0.0
        # load lambda config
        lambdas = {}
        for k, v in lambda_config.items():
            if isinstance(v, dict):
                start = v['start']
                end = v['end']
                iterations = v['iterations']
                gamma = (end / start) ** (1 / iterations)
                l_type = 'exponential_decay' if start > end else 'exponential_growth'
                lambdas[k] = {'gamma': nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=False),
                              'end': nn.Parameter(torch.tensor(end, dtype=torch.float32), requires_grad=False),
                              'value': nn.Parameter(torch.tensor(start, dtype=torch.float32), requires_grad=False),
                              'type': l_type}
            else:
                lambdas[k] = {'value': nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False),
                              'type': 'constant'}

        self.lambdas = nn.ParameterDict(lambdas)

        self.correction_modules = nn.ModuleDict(correction_modules)
        self.calibration_modules = nn.ModuleDict(calibration_modules)
        self.scaling_modules = nn.ModuleDict(scaling_modules)
        self.alignment_modules = nn.ModuleDict(alignment_modules)
        self.background_modules = nn.ModuleDict(background_modules)
        self.mse_loss = nn.MSELoss()

        # solar wind
        velocity_min = (100.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_min = nn.Parameter(torch.tensor(velocity_min, dtype=torch.float32), requires_grad=False)
        velocity_max = (1000.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_max = nn.Parameter(torch.tensor(velocity_max, dtype=torch.float32), requires_grad=False)
        velocity_avg = (300.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_avg = nn.Parameter(torch.tensor(velocity_avg, dtype=torch.float32), requires_grad=False)

        # radial weighting
        self.min_radius_weight = nn.Parameter(torch.tensor(1.0 / Rs_per_ds, dtype=torch.float32), requires_grad=False)
        self.max_radius_weight = nn.Parameter(torch.tensor(10.0 / Rs_per_ds, dtype=torch.float32), requires_grad=False)

        print(f'Velocity min: {velocity_min}, max: {velocity_max}')
        drop_off_distance = (1 * u.AU).to_value(u.R_sun) / Rs_per_ds
        self.drop_off_distance = nn.Parameter(torch.tensor(drop_off_distance, dtype=torch.float32), requires_grad=False)

    @staticmethod
    def _normalize_with_scaling_mask(image: torch.Tensor, batch: dict) -> torch.Tensor:
        scaling_mask = batch.get('scaling_mask', None)
        if scaling_mask is None:
            return image
        # no clamping here, as scaling mask should already be properly regularized
        return image / scaling_mask

    def training_step(self, batch, batch_nb):
        dataset_batch = {k: v for k, v in batch.items() if k != 'random'}

        # apply alignment modules
        for k in dataset_batch.keys():
            instrument_key = batch[k]['instrument']

            if instrument_key in self.alignment_modules:
                b_rays = batch[k]['rays']
                time = batch[k]['time']
                aligned_rays = self.alignment_modules[instrument_key](b_rays, time)
                batch[k]['rays'] = aligned_rays

        rendering_out = self.rendering(dataset_batch)

        model_out = rendering_out['model_out']

        instrument_tB_image_diff = []
        instrument_pB_image_diff = []
        instrument_ratio_diff = []
        correction_losses = []

        for k in dataset_batch.keys():
            instrument_key = batch[k]['instrument']
            image_coords = batch[k]['image_coords']
            hpc_coords = batch[k]['hpc_coords']
            time = batch[k]['time']
            rays_d = batch[k]['rays'][..., 1, :]

            model_image = model_out[k]['image']
            target_image = dataset_batch[k]['image']

            if instrument_key in self.background_modules:
                background = self.background_modules[instrument_key](rays_d)
                model_image = model_image + background

                if self.lambdas['star_background']['value'] > 0.0:
                    # L1 loss to encourage sparse star background
                    star_background_loss = torch.abs(background)
                    correction_losses.append({'star_background': star_background_loss})

            if instrument_key in self.correction_modules:
                model_image, correction = self.correction_modules[instrument_key](model_image, image_coords, hpc_coords,
                                                                                  time)
                # compute correction losses
                correction_losses.append(self.get_correction_loss(correction))
            if instrument_key in self.calibration_modules:
                model_image = self.calibration_modules[instrument_key](model_image)

            tB_nan_mask = ~torch.isnan(target_image[..., 0])
            pB_nan_mask = ~torch.isnan(target_image[..., 1])
            ratio_nan_mask = tB_nan_mask & pB_nan_mask

            # compute polarization ratios
            ratio_target_image = target_image[ratio_nan_mask, 1] / (target_image[ratio_nan_mask, 0] + 1e-8)
            ratio_model_image = model_image[ratio_nan_mask, 1] / (model_image[ratio_nan_mask, 0] + 1e-8)

            # clip ratios to prevent extreme values from dominating the loss
            ratio_target_image = torch.clamp(ratio_target_image, 0.0, 1.0)
            ratio_model_image = torch.clamp(ratio_model_image, 0.0, 1.0)

            # scale images
            image_scaling = self.scaling_modules[instrument_key]
            model_image = self._normalize_with_scaling_mask(model_image, dataset_batch[k])
            target_image = self._normalize_with_scaling_mask(target_image, dataset_batch[k])
            scaled_model_image = image_scaling(model_image)
            scaled_target_image = image_scaling(target_image)

            # backpropagation
            # optimize model
            tB_image_diff = (scaled_model_image[tB_nan_mask, 0] - scaled_target_image[tB_nan_mask, 0]).pow(2)
            pB_image_diff = (scaled_model_image[pB_nan_mask, 1] - scaled_target_image[pB_nan_mask, 1]).pow(2)
            ratio_diff = (ratio_model_image - ratio_target_image).pow(2)

            instrument_tB_image_diff.append(tB_image_diff)
            instrument_pB_image_diff.append(pB_image_diff)
            instrument_ratio_diff.append(ratio_diff)

        tB_image_loss = torch.cat(instrument_tB_image_diff)
        tB_image_loss = torch.zeros((1,), dtype=torch.float32, device=instrument_pB_image_diff.device) if tB_image_loss.shape[0] == 0 else tB_image_loss.mean()
        #
        pB_image_loss = torch.cat(instrument_pB_image_diff)
        pB_image_loss = torch.zeros_like(tB_image_loss) if pB_image_loss.shape[0] == 0 else pB_image_loss.mean()
        image_loss = (tB_image_loss + pB_image_loss)
        #
        ratio_loss = torch.cat(instrument_ratio_diff)
        ratio_loss = torch.zeros_like(tB_image_loss) if ratio_loss.shape[0] == 0 else ratio_loss.mean()

        assert torch.isnan(image_loss).sum() == 0, 'Invalid loss detected: image_loss'
        assert torch.isnan(ratio_loss).sum() == 0, 'Invalid loss detected: ratio_loss'

        loss = self.lambdas['image']['value'] * image_loss + self.lambdas['ratio']['value'] * ratio_loss

        with torch.no_grad():
            psnr = -10. * torch.log10(image_loss)

        log_values = {'image': image_loss, 'psnr': psnr,
                      'ratio': ratio_loss}

        # add correction losses/regularizations
        correction_keys = set([k for correction_dict in correction_losses for k in correction_dict.keys()])
        correction_losses = {k: torch.cat([cl[k] for cl in correction_losses if k in cl]).mean()
                             for k in correction_keys}
        for k, v in correction_losses.items():
            loss += self.lambdas[k]['value'] * v
            log_values[k] = v

        if self.lambdas['calibration']['value'] > 0.0 and len(self.calibration_modules) > 0:
            calibration_loss = torch.stack([
                (torch.exp(module.calibration) - 1.0).pow(2).mean()
                for module in self.calibration_modules.values()
            ]).mean()
            loss += self.lambdas['calibration']['value'] * calibration_loss
            log_values['calibration'] = calibration_loss

        if 'random' in batch:
            query_points = batch['random']['coords']
            query_points.requires_grad = True

            r = torch.norm(query_points[:, :3], dim=-1)
            radial_weight = torch.clamp(
                (r - self.min_radius_weight) / (self.max_radius_weight - self.min_radius_weight),
                min=0.0, max=1.0).pow(2)

            model_out = self.model(query_points)

            rho = model_out['rho']
            log_rho = model_out['log_rho']
            v = model_out['v']
            # continuity_loss = self.compute_continuity_loss(rho, v, query_points)
            continuity_loss = self.compute_log_continuity_loss(log_rho, v, query_points)
            continuity_loss = continuity_loss.mean()

            loss += self.lambdas['continuity']['value'] * continuity_loss
            log_values['continuity'] = continuity_loss

            # velocity regularization
            r_hat = query_points[:, :3] / (torch.norm(query_points[:, :3], dim=-1, keepdim=True) + 1e-7)
            v_radial = (v * r_hat).sum(dim=-1)
            min_v = torch.clip(v_radial - self.velocity_min, max=0).pow(2)
            max_v = torch.clip(v_radial - self.velocity_max, min=0).pow(2)
            velocity_loss = min_v + max_v
            velocity_loss = (velocity_loss * radial_weight).sum() / (radial_weight.sum() + 1e-7)
            log_values['velocity'] = velocity_loss
            loss += self.lambdas['velocity']['value'] * velocity_loss

            # radial regularization
            normalization = torch.norm(query_points[:, :3], dim=-1) * torch.norm(v, dim=-1) + 1e-7
            radial_loss = torch.norm(torch.cross(v, query_points[:, :3], dim=-1), dim=-1) / normalization
            radial_loss = radial_loss.pow(2)
            radial_loss = (radial_loss * radial_weight).sum() / (radial_weight.sum() + 1e-7)
            log_values['radial'] = radial_loss
            loss += self.lambdas['radial']['value'] * radial_loss

            assert torch.isnan(continuity_loss).sum() == 0, 'Invalid loss detected: continuity_loss'
            assert torch.isnan(velocity_loss).sum() == 0, 'Invalid loss detected: velocity_loss'
            assert torch.isnan(radial_loss).sum() == 0, 'Invalid loss detected: radial_loss'

        assert torch.isnan(loss).sum() == 0, 'Invalid loss detected: loss'
        # log results to WANDB
        self.log("loss", loss)
        self.log_dict({f'train.{k}': v for k, v in log_values.items()})

        return loss

    def get_correction_loss(self, correction):
        correction_losses = {}
        if self.lambdas['calibration']['value'] > 0.0 and 'calibration' in correction:
            calibration = correction['calibration']
            # prefer temporal calibration close to 1
            calibration_loss = (calibration - 1.0).pow(2)
            correction_losses['calibration'] = calibration_loss
        if self.lambdas['f_corona']['value'] > 0.0 and 'f_corona' in correction:
            f_corona = correction['f_corona']
            # prefer smaller coronal brightness
            f_corona_loss = f_corona.pow(2)
            correction_losses['f_corona'] = f_corona_loss
        if self.lambdas['transmission']['value'] > 0.0 and 'transmission' in correction:
            transmission = correction['transmission']
            # prefer transmission close to 1
            transmission_loss = (transmission - 1.0).pow(2)
            correction_losses['transmission'] = transmission_loss
        if self.lambdas['calibration_gain']['value'] > 0.0 and 'calibration_gain' in correction:
            calibration_gain = correction['calibration_gain']
            # prefer calibration gain close to 1
            calibration_gain_loss = (calibration_gain - 1.0).pow(2)
            correction_losses['calibration_gain'] = calibration_gain_loss
        if self.lambdas['calibration_offset']['value'] > 0.0 and 'calibration_offset' in correction:
            calibration_offset = correction['calibration_offset']
            # prefer small calibration offset
            calibration_offset_loss = calibration_offset.pow(2)
            correction_losses['calibration_offset'] = calibration_offset_loss
        if self.lambdas['calibration_scalar']['value'] > 0.0 and 'calibration_scalar' in correction:
            calibration_scalar = correction['calibration_scalar']
            # prefer calibration scalar close to 1
            calibration_scalar_loss = (calibration_scalar - 1.0).pow(2)
            correction_losses['calibration_scalar'] = calibration_scalar_loss
        if self.lambdas['pB_mul']['value'] > 0.0 and 'pB_mul' in correction:
            pB_mul = correction['pB_mul']
            # prefer pB multiplicative correction close to 1
            pB_mul_loss = (pB_mul - 1.0).pow(2)
            correction_losses['pB_mul'] = pB_mul_loss
        if self.lambdas['tB_mul']['value'] > 0.0 and 'tB_mul' in correction:
            tB_mul = correction['tB_mul']
            # prefer tB multiplicative correction close to 1
            tB_mul_loss = (tB_mul - 1.0).pow(2)
            correction_losses['tB_mul'] = tB_mul_loss
        if self.lambdas['pB_add']['value'] > 0.0 and 'pB_add' in correction:
            pB_add = correction['pB_add']
            # prefer small pB additive correction
            pB_add_loss = pB_add.pow(2)
            correction_losses['pB_add'] = pB_add_loss
        if self.lambdas['tB_add']['value'] > 0.0 and 'tB_add' in correction:
            tB_add = correction['tB_add']
            # prefer small tB additive correction
            tB_add_loss = tB_add.pow(2)
            correction_losses['tB_add'] = tB_add_loss
        if self.lambdas['pB_add_mean']['value'] > 0.0 and 'pB_add' in correction:
            pB_add = correction['pB_add']
            # prefer zero-mean pB additive correction over the sample
            pB_add_mean_loss = pB_add.mean().pow(2).reshape(1, 1)
            correction_losses['pB_add_mean'] = pB_add_mean_loss
        if self.lambdas['tB_add_mean']['value'] > 0.0 and 'tB_add' in correction:
            tB_add = correction['tB_add']
            # prefer zero-mean tB additive correction over the sample
            tB_add_mean_loss = tB_add.mean().pow(2).reshape(1, 1)
            correction_losses['tB_add_mean'] = tB_add_mean_loss
        if self.lambdas['pB_straylight']['value'] > 0.0 and 'pB_straylight' in correction:
            pB_straylight = correction['pB_straylight']
            # prefer small positive pB straylight correction
            pB_straylight_loss = pB_straylight.pow(2)
            correction_losses['pB_straylight'] = pB_straylight_loss
        if self.lambdas['tB_straylight']['value'] > 0.0 and 'tB_straylight' in correction:
            tB_straylight = correction['tB_straylight']
            # prefer small positive tB straylight correction
            tB_straylight_loss = tB_straylight.pow(2)
            correction_losses['tB_straylight'] = tB_straylight_loss
        return correction_losses

    def compute_continuity_loss(self, rho, v, query_points):
        rho_jac_matrix = jacobian(rho, query_points)
        dRho_dx = rho_jac_matrix[:, 0, 0]
        dRho_dy = rho_jac_matrix[:, 0, 1]
        dRho_dz = rho_jac_matrix[:, 0, 2]
        dRho_dt = rho_jac_matrix[:, 0, 3]

        rho_v = rho * v
        v_jac_matrix = jacobian(rho_v, query_points)
        dRhoVx_dx = v_jac_matrix[:, 0, 0]
        dRhoVy_dy = v_jac_matrix[:, 1, 1]
        dRhoVz_dz = v_jac_matrix[:, 2, 2]

        div_rho_v = dRhoVx_dx + dRhoVy_dy + dRhoVz_dz
        continuity_eq = dRho_dt + div_rho_v

        loss = continuity_eq.pow(2)
        radial_distance = torch.norm(query_points[..., :3], dim=-1)
        # compensate for the radial drop-off
        loss = loss * radial_distance.pow(4)
        # normalize density
        loss = loss / (rho * radial_distance.pow(2) + 1e-6).pow(2).mean()

        return loss

    def compute_log_continuity_loss(self, log_rho, v, query_points):
        rho_jac_matrix = jacobian(log_rho, query_points)
        dlogRho_dx = rho_jac_matrix[:, 0, 0]
        dlogRho_dy = rho_jac_matrix[:, 0, 1]
        dlogRho_dz = rho_jac_matrix[:, 0, 2]
        dlogRho_dt = rho_jac_matrix[:, 0, 3]

        v_jac_matrix = jacobian(v, query_points)
        dVx_dx = v_jac_matrix[:, 0, 0]
        dVy_dy = v_jac_matrix[:, 1, 1]
        dVz_dz = v_jac_matrix[:, 2, 2]

        div_V = (dVx_dx + dVy_dy + dVz_dz)
        grad_logRho = torch.stack([dlogRho_dx, dlogRho_dy, dlogRho_dz], -1)
        v_dot_grad_logRho = (v * grad_logRho).sum(-1)
        continuity_eq = dlogRho_dt + div_V + v_dot_grad_logRho

        loss = continuity_eq.pow(2)
        return loss

    def validation_step(self, batch, batch_idx, *args):
        """
        Validation routing is fully controlled by batch['render_mode'] (injected by wrapper dataset).
        """
        dataloader_idx = args[0] if len(args) > 0 else 0
        dataset_key = self.validation_dataset_mapping[dataloader_idx]

        if "render_mode" not in batch:
            raise KeyError(
                "Missing 'render_mode' in validation batch. Wrap validation datasets with RenderModeDataset."
            )

        mode = RenderMode(int(batch["render_mode"].view(-1)[0].item()))

        if mode == RenderMode.INSTRUMENT:
            return self._val_instrument(batch, dataset_key)

        if mode == RenderMode.BACKGROUND:
            return self._val_background_only(batch)

        if mode == RenderMode.REFERENCE:
            return self._val_reference(batch)

        if mode == RenderMode.QUERY_POINTS:
            return self._val_query_points(batch)

        raise ValueError(f"Unknown render_mode={mode}")

    def _val_instrument(self, batch, dataset_key: str):
        instrument_key = batch["instrument"]
        image_coords = batch["image_coords"]
        hpc_coords = batch["hpc_coords"]
        image = batch["image"]
        time = batch["time"]
        rays_d = batch["rays"][..., 1, :]

        # alignment (optional)
        if instrument_key in self.alignment_modules:
            b_rays = batch["rays"]
            batch["rays"] = self.alignment_modules[instrument_key](b_rays, time)
            rays_d = batch["rays"][..., 1, :]

        rendering_out = self.rendering({dataset_key: batch})
        model_out = rendering_out["model_out"][dataset_key]
        model_image = model_out["image"]

        # background (optional)
        background = None
        if instrument_key in self.background_modules:
            background = self.background_modules[instrument_key](rays_d)
            model_image = model_image + background

        # correction (optional)
        corrections = None
        if instrument_key in self.correction_modules:
            model_image, corrections = self.correction_modules[instrument_key](
                model_image, image_coords, hpc_coords, time
            )

        # calibration (optional)
        if instrument_key in self.calibration_modules:
            model_image = self.calibration_modules[instrument_key](model_image)

        # ratios (safe for NaNs)
        target_ratio = image[..., 1:2] / (image[..., 0:1] + 1e-8)
        model_ratio = model_image[..., 1:2] / (model_image[..., 0:1] + 1e-8)

        # clip ratios to prevent extreme values from dominating the loss
        # >1 is unphysical, but can be caused by correction/calibration
        target_ratio = torch.clamp(target_ratio, 0.0, 2.0)
        model_ratio = torch.clamp(model_ratio, 0.0, 2.0)

        # scale images consistently
        image_scaling = self.scaling_modules[instrument_key]
        image = self._normalize_with_scaling_mask(image, batch)
        model_image = self._normalize_with_scaling_mask(model_image, batch)
        target_image = image_scaling(image)
        model_image = image_scaling(model_image)

        result = {
            "target_image": target_image,
            "model_image": model_image,
            "model_ratio": model_ratio,
            "target_ratio": target_ratio,
            "density": model_out.get("density", None),
            "distance_from_sun": model_out.get("distance_from_sun", None),
            "distance_from_obs": model_out.get("distance_from_obs", None),
            "distance": model_out.get("distance", None),
        }

        # attach corrections/background if present
        if corrections is not None:
            for k, v in corrections.items():
                result[f"correction.{k}"] = v
        if background is not None:
            result["background"] = background

        # prune None values (keeps callbacks simpler)
        return {k: v for k, v in result.items() if v is not None}

    def _val_query_points(self, batch):
        query_points = batch["query_points"]
        model_out = self.model(query_points)
        result = {
            "rho_pred": model_out["rho"],
            "v_pred": model_out["v"],
            "query_points": query_points,
        }
        if "spherical_coords" in batch:
            result["spherical_coords"] = batch["spherical_coords"]
        # include meta if present (optional, helpful for plotting)
        for k in list(batch.keys()):
            if k.startswith("meta."):
                result[k] = batch[k]
        return result

    def _val_reference(self, batch):
        query_points = batch["query_points"]
        rho_true = batch["rho"]
        model_out = self.model(query_points)
        result = {
            "rho_true": rho_true,
            "rho_pred": model_out["rho"],
            "v_pred": model_out["v"],
            "query_points": query_points,
        }
        if "spherical_coords" in batch:
            result["spherical_coords"] = batch["spherical_coords"]
        for k in list(batch.keys()):
            if k.startswith("meta."):
                result[k] = batch[k]
        return result

    def _val_background_only(self, batch):
        instrument_key = batch["instrument"]
        rays_d = batch["rays"][..., 1, :]

        if instrument_key in self.alignment_modules:
            b_rays = batch["rays"]
            time = batch["time"]
            batch["rays"] = self.alignment_modules[instrument_key](b_rays, time)
            rays_d = batch["rays"][..., 1, :]

        if instrument_key in self.background_modules:
            background = self.background_modules[instrument_key](rays_d)
            return {"background": background}

        return {}

    def on_train_batch_end(self, *args, **kwargs):
        self.model.step(self.global_step)
        # update lambda values
        for k, v in self.lambdas.items():
            if v['type'] == 'exponential_decay':
                new_value = v['value'] * v['gamma']
                if new_value <= v['end']:
                    new_value = v['end']
                v['value'] = new_value
                self.log(f'lambda_{k}', float(v['value'].detach().cpu().numpy()), sync_dist=True)
            if v['type'] == 'exponential_growth':
                new_value = v['value'] * v['gamma']
                if new_value >= v['end']:
                    new_value = v['end']
                v['value'] = new_value
                self.log(f'lambda_{k}', float(v['value'].detach().cpu().numpy()), sync_dist=True)
            if v['type'] == 'constant':
                pass  # no change required, no logging
        # log instrument scaling
        scaling = {f'instrument_calibration.{k}': float(torch.exp(m.calibration).detach().cpu().numpy())
                   for k, m in self.calibration_modules.items()}
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'current_alpha'):
            scaling['dynamic_alpha'] = float(self.model.model.current_alpha.detach().cpu().numpy())
        self.log_dict(scaling, sync_dist=True)
        # call super method
        super().on_train_batch_end(*args, **kwargs)


def save_thomson_sunerf(sunerf: ThomsonSuNeRFModule, data_module: BaseDataModule, save_path,
                        msb_norm=None, msb=None, sigma_ne=None):
    output_path = '/'.join(save_path.split('/')[0:-1])
    os.makedirs(output_path, exist_ok=True)
    first_rendering_module = next(iter(sunerf.rendering_modules.values()))
    state = {
        # sunerf  rendering module
        'rendering': sunerf.rendering,
        # data infor
        'data_config': data_module.config,
        # data scaling
        'Rs_per_ds': data_module.Rs_per_ds,
        'seconds_per_dt': data_module.seconds_per_dt,
        'ref_date': data_module.ref_date,
        'thomson_normalization': {
            'c0': float(first_rendering_module.C_0.detach().cpu().numpy()),
            'msb_norm': msb_norm,
            'msb': msb,
            'sigma_ne': sigma_ne,
        }
    }
    torch.save(state, save_path)
