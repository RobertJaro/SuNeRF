import os
import copy

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
                 shuffle_config=None,
                 light_travel_time=True,
                 physics_update_interval=1,
                 insitu_jitter_std_rsun=0.0,
                 ballistic_acceleration_limit_kms2=1.0,
                 **kwargs):
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
            if instrument_type == 'insitu':
                if calibration:
                    calibration_config = {} if isinstance(calibration, bool) else calibration
                    calibration_config = calibration_config.copy()
                    calibration_trainable = calibration_config.pop('trainable', True)
                    calibration_module = CalibrationModule(**calibration_config)
                    calibration_module.requires_grad_(calibration_trainable)
                    calibration_modules[instrument_key] = calibration_module
                continue
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
                correction_config = correction_config.copy()
                correction_trainable = correction_config.pop('trainable', True)
                correction_module = CorrectionModule(**correction_config)
                correction_module.requires_grad_(correction_trainable)
                correction_modules[instrument_key] = correction_module
            # calibration module
            if calibration:
                calibration_config = {} if isinstance(calibration, bool) else calibration
                calibration_config = calibration_config.copy()
                calibration_trainable = calibration_config.pop('trainable', True)
                calibration_module = CalibrationModule(**calibration_config)
                calibration_module.requires_grad_(calibration_trainable)
                calibration_modules[instrument_key] = calibration_module
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
                                         seconds_per_dt=seconds_per_dt,
                                         light_travel_time=light_travel_time,
                                         sampling_config=sampling_config, shuffle_config=shuffle_config)

        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        self.rendering_modules = rendering_modules
        self.model = model

        # define lambda values
        lambda_config = {'image': 1.0,
                         'ratio': 1.0,
                         'continuity': 1e-3,
                         'ballistic': 1e-4,
                         'radial': 1e-2,
                         'velocity': 1e-3} if lambda_config is None else lambda_config
        # check lambda config
        available_lambdas = ['image', 'ratio', 'continuity', 'ballistic', 'radial', 'velocity',
                             'insitu_density', 'insitu_velocity_radial',
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
                l_type = v.get('type', None)
                if l_type is None or l_type == 'exponential':
                    iterations = v['iterations']
                    gamma = (end / start) ** (1 / iterations)
                    l_type = 'exponential_decay' if start > end else 'exponential_growth'
                    lambdas[k] = {'gamma': nn.Parameter(torch.tensor(gamma, dtype=torch.float32), requires_grad=False),
                                  'end': nn.Parameter(torch.tensor(end, dtype=torch.float32), requires_grad=False),
                                  'value': nn.Parameter(torch.tensor(start, dtype=torch.float32), requires_grad=False),
                                  'type': l_type}
                elif l_type == 'step':
                    steps = v['steps']
                    lambdas[k] = {'steps': nn.Parameter(torch.tensor(int(steps), dtype=torch.int64), requires_grad=False),
                                  'end': nn.Parameter(torch.tensor(end, dtype=torch.float32), requires_grad=False),
                                  'value': nn.Parameter(torch.tensor(start, dtype=torch.float32), requires_grad=False),
                                  'type': l_type}
                else:
                    raise ValueError(f"Invalid lambda schedule type: {l_type}, must be in ['exponential', 'step']")
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
        self.physics_update_interval = max(1, int(physics_update_interval))
        if insitu_jitter_std_rsun < 0:
            raise ValueError("insitu_jitter_std_rsun must be non-negative.")
        self.insitu_jitter_std = float(insitu_jitter_std_rsun) / float(Rs_per_ds)

        # solar wind
        velocity_min = (100.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_min = nn.Parameter(torch.tensor(velocity_min, dtype=torch.float32), requires_grad=False)
        velocity_max = (1000.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_max = nn.Parameter(torch.tensor(velocity_max, dtype=torch.float32), requires_grad=False)
        velocity_avg = (300.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_avg = nn.Parameter(torch.tensor(velocity_avg, dtype=torch.float32), requires_grad=False)
        velocity_loss_scale = (100.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt
        self.velocity_loss_scale = nn.Parameter(torch.tensor(velocity_loss_scale, dtype=torch.float32),
                                                requires_grad=False)
        if ballistic_acceleration_limit_kms2 <= 0:
            raise ValueError("ballistic_acceleration_limit_kms2 must be positive.")
        self.ballistic_acceleration_limit = (
            (ballistic_acceleration_limit_kms2 * u.km / u.s ** 2).to_value(u.R_sun / u.s ** 2)
            / Rs_per_ds * seconds_per_dt ** 2
        )
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

    @staticmethod
    def _mean_or_zero(values, device):
        values = [v for v in values if v.numel() > 0]
        if len(values) == 0:
            return torch.zeros((1,), dtype=torch.float32, device=device)
        return torch.cat(values).mean()

    def _jitter_insitu_query_points(self, query_points: torch.Tensor) -> torch.Tensor:
        """Apply training-only spatial Gaussian jitter to in-situ coordinates."""
        if self.insitu_jitter_std == 0.0:
            return query_points
        spatial = query_points[..., :3]
        jittered_spatial = spatial + torch.randn_like(spatial) * self.insitu_jitter_std
        return torch.cat([jittered_spatial, query_points[..., 3:]], dim=-1)

    def _apply_alignment(self, batch: dict) -> dict:
        instrument_key = batch['instrument']
        if instrument_key in self.alignment_modules:
            batch['rays'] = self.alignment_modules[instrument_key](batch['rays'], batch['time'])
        return batch

    def _apply_image_modules(self, batch: dict, model_image: torch.Tensor,
                             correction_losses: list | None = None,
                             collect_regularization: bool = False):
        instrument_key = batch['instrument']
        rays_d = batch['rays'][..., 1, :]
        aux = {}

        if instrument_key in self.background_modules:
            background = self.background_modules[instrument_key](rays_d)
            model_image = model_image + background
            aux['background'] = background
            if collect_regularization and self.lambdas['star_background']['value'] > 0.0 and correction_losses is not None:
                correction_losses.append({'star_background': torch.abs(background)})

        if instrument_key in self.correction_modules:
            model_image, correction = self.correction_modules[instrument_key](
                model_image, batch['image_coords'], batch['hpc_coords'], batch['time']
            )
            aux['correction'] = correction
            if collect_regularization and correction_losses is not None:
                correction_losses.append(self.get_correction_loss(correction))

        if instrument_key in self.calibration_modules:
            model_image = self.calibration_modules[instrument_key](model_image)

        return model_image, aux

    def training_step(self, batch, batch_nb):
        insitu_batch = {
            k: v for k, v in batch.items()
            if k != 'random' and isinstance(v, dict) and 'query_points' in v and 'density' in v
        }
        insitu_query_points = {
            k: self._jitter_insitu_query_points(v['query_points'])
            for k, v in insitu_batch.items()
        }
        dataset_batch = {
            k: v for k, v in batch.items()
            if k != 'random' and k not in insitu_batch
        }
        device = self.drop_off_distance.device

        render_batch = {}
        image_loss_batch = {}
        for k, image_batch in dataset_batch.items():
            image_batch = self._apply_alignment(image_batch)
            render_batch[k] = image_batch
            image_loss_batch[k] = image_batch
        rendering_out = self.rendering(render_batch) if len(render_batch) > 0 else {'model_out': {}}
        model_out = rendering_out['model_out']

        instrument_tB_image_diff = []
        instrument_pB_image_diff = []
        instrument_ratio_diff = []
        correction_losses = []
        for k, image_batch in image_loss_batch.items():
            instrument_key = image_batch['instrument']

            model_image = model_out[k]['image']
            target_image = image_batch['image']
            model_image, _ = self._apply_image_modules(
                image_batch,
                model_image,
                correction_losses=correction_losses,
                collect_regularization=k in dataset_batch,
            )
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
            model_image = self._normalize_with_scaling_mask(model_image, image_batch)
            target_image = self._normalize_with_scaling_mask(target_image, image_batch)
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

        tB_image_loss = self._mean_or_zero(instrument_tB_image_diff, device)
        pB_image_loss = self._mean_or_zero(instrument_pB_image_diff, device)
        image_loss = (tB_image_loss + pB_image_loss)
        ratio_loss = self._mean_or_zero(instrument_ratio_diff, device)
        assert torch.isnan(image_loss).sum() == 0, 'Invalid loss detected: image_loss'
        assert torch.isnan(ratio_loss).sum() == 0, 'Invalid loss detected: ratio_loss'
        loss = (self.lambdas['image']['value'] * image_loss +
                self.lambdas['ratio']['value'] * ratio_loss)

        with torch.no_grad():
            psnr = -10. * torch.log10(image_loss)

        log_values = {'image': image_loss, 'psnr': psnr,
                      'ratio': ratio_loss}

        if len(insitu_batch) > 0:
            insitu_density_losses = []
            insitu_velocity_losses = []
            for k, insitu in insitu_batch.items():
                instrument_key = insitu.get('instrument', k)
                query_points = insitu_query_points[k]
                model_out_insitu = self.model(query_points)

                rho_pred = torch.clamp(model_out_insitu['rho'], min=1e-12)
                if instrument_key in self.calibration_modules:
                    rho_pred = self.calibration_modules[instrument_key](rho_pred)
                rho_target = torch.clamp(insitu['density'], min=1e-12)
                density_mask = insitu.get('has_density', torch.isfinite(insitu['density'])).bool()
                density_mask = density_mask & torch.isfinite(rho_target)
                if density_mask.any():
                    density_loss = (torch.log(rho_pred[density_mask]) - torch.log(rho_target[density_mask])).pow(2)
                    insitu_density_losses.append(density_loss)

                if 'velocity_radial' in insitu and 'r_hat' in insitu:
                    v_pred = model_out_insitu['v']
                    v_radial_pred = (v_pred * insitu['r_hat']).sum(dim=-1, keepdim=True)
                    velocity_mask = insitu.get('has_velocity', torch.isfinite(insitu['velocity_radial'])).bool()
                    velocity_mask = velocity_mask & torch.isfinite(insitu['velocity_radial'])
                    if velocity_mask.any():
                        velocity_diff = (v_radial_pred[velocity_mask] - insitu['velocity_radial'][velocity_mask]) / self.velocity_loss_scale
                        insitu_velocity_losses.append(velocity_diff.pow(2))

            if len(insitu_density_losses) > 0:
                insitu_density_loss = torch.cat(insitu_density_losses).mean()
                loss += self.lambdas['insitu_density']['value'] * insitu_density_loss
                log_values['insitu_density'] = insitu_density_loss
            if len(insitu_velocity_losses) > 0:
                insitu_velocity_loss = torch.cat(insitu_velocity_losses).mean()
                loss += self.lambdas['insitu_velocity_radial']['value'] * insitu_velocity_loss
                log_values['insitu_velocity_radial'] = insitu_velocity_loss

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

        physics_loss_keys = ('continuity', 'ballistic', 'velocity', 'radial')
        active_physics = {key: self.lambdas[key]['value'] > 0.0 for key in physics_loss_keys}
        has_active_physics_loss = any(active_physics.values())
        train_step = self.global_step + 1
        should_update_physics = train_step % self.physics_update_interval == 0
        if 'random' in batch and has_active_physics_loss and should_update_physics:
            random_query_points = batch['random']['coords']
            n_random_points = random_query_points.shape[0]
            physics_query_points = [random_query_points]
            if active_physics['continuity']:
                physics_query_points.extend(
                    insitu_query_points[k] for k in insitu_batch
                )
            query_points = torch.cat(physics_query_points, dim=0)
            query_points.requires_grad_(True)

            random_points = query_points[:n_random_points]
            model_out = self.model(query_points)
            v = model_out['v']

            jacobian_fields = {}
            if active_physics['continuity']:
                jacobian_fields['log_rho'] = model_out['log_rho']
            if active_physics['continuity'] or active_physics['ballistic']:
                jacobian_fields['v'] = v
            jacobian_matrices = self.compute_physics_jacobians(query_points, **jacobian_fields)
            radial_weight = None
            if active_physics['velocity'] or active_physics['radial']:
                radial_weight = self.compute_physics_radial_weight(random_points)

            if active_physics['continuity']:
                continuity_point_loss, continuity_terms = self.compute_log_continuity_loss(
                    v, jacobian_matrices['log_rho'], jacobian_matrices['v']
                )
                continuity_loss = continuity_point_loss.mean()
                loss += self.lambdas['continuity']['value'] * continuity_loss
                log_values['continuity.loss'] = continuity_loss
                for term_name, term_value in continuity_terms.items():
                    log_values[f'continuity.{term_name}'] = term_value.pow(2).mean().sqrt()
                assert not torch.isnan(continuity_loss).any(), 'Invalid loss detected: continuity_loss'

            if active_physics['ballistic']:
                ballistic_point_loss = self.compute_ballistic_loss(v, jacobian_matrices['v'])
                ballistic_loss = ballistic_point_loss[:n_random_points].mean()
                loss += self.lambdas['ballistic']['value'] * ballistic_loss
                log_values['ballistic.loss'] = ballistic_loss
                assert not torch.isnan(ballistic_loss).any(), 'Invalid loss detected: ballistic_loss'

            if active_physics['velocity'] or active_physics['radial']:
                random_v = v[:n_random_points]

                if active_physics['velocity']:
                    r_hat = random_points[:, :3] / (
                        torch.norm(random_points[:, :3], dim=-1, keepdim=True) + 1e-7
                    )
                    v_radial = (random_v * r_hat).sum(dim=-1)
                    min_v = torch.clip(v_radial - self.velocity_min, max=0).pow(2)
                    max_v = torch.clip(v_radial - self.velocity_max, min=0).pow(2)
                    velocity_loss = ((min_v + max_v) * radial_weight).sum() / (radial_weight.sum() + 1e-7)
                    log_values['velocity'] = velocity_loss
                    loss += self.lambdas['velocity']['value'] * velocity_loss
                    assert not torch.isnan(velocity_loss).any(), 'Invalid loss detected: velocity_loss'

                if active_physics['radial']:
                    normalization = (
                        torch.norm(random_points[:, :3], dim=-1) * torch.norm(random_v, dim=-1) + 1e-7
                    )
                    radial_loss = torch.norm(
                        torch.cross(random_v, random_points[:, :3], dim=-1), dim=-1
                    ) / normalization
                    radial_loss = (radial_loss.pow(2) * radial_weight).sum() / (radial_weight.sum() + 1e-7)
                    log_values['radial'] = radial_loss
                    loss += self.lambdas['radial']['value'] * radial_loss
                    assert not torch.isnan(radial_loss).any(), 'Invalid loss detected: radial_loss'
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

    @staticmethod
    def compute_physics_jacobians(query_points, **fields):
        """Compute each requested field Jacobian once with respect to the shared coordinates."""
        return {name: jacobian(field, query_points) for name, field in fields.items()}

    def compute_physics_radial_weight(self, query_points):
        radius = torch.norm(query_points[:, :3], dim=-1)
        return torch.clamp(
            (radius - self.min_radius_weight) / (self.max_radius_weight - self.min_radius_weight),
            min=0.0, max=1.0
        ).pow(2)

    @staticmethod
    def compute_log_continuity_loss(v, log_rho_jacobian, velocity_jacobian):
        grad_log_rho = log_rho_jacobian[:, 0, :3]
        dlog_rho_dt = log_rho_jacobian[:, 0, 3]
        div_v = torch.diagonal(velocity_jacobian[:, :, :3], dim1=1, dim2=2).sum(dim=-1)

        v_dot_grad_log_rho = (v * grad_log_rho).sum(dim=-1)
        continuity_eq = dlog_rho_dt + div_v + v_dot_grad_log_rho

        loss = continuity_eq.pow(2)
        continuity_terms = {
            'dlog_rho_dt': dlog_rho_dt,
            'div_v': div_v,
            'v_dot_grad_log_rho': v_dot_grad_log_rho,
            'residual': continuity_eq,
        }
        return loss, continuity_terms

    def compute_ballistic_loss(self, v, velocity_jacobian):
        """Penalize material acceleration only above the configured magnitude limit."""
        dv_dt = velocity_jacobian[:, :, 3]
        advective_acceleration = torch.einsum('ni,nji->nj', v, velocity_jacobian[:, :, :3])
        acceleration_magnitude = torch.norm(dv_dt + advective_acceleration, dim=-1)
        acceleration_excess = torch.relu(acceleration_magnitude - self.ballistic_acceleration_limit)
        return (acceleration_excess / self.ballistic_acceleration_limit).pow(2)

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
        image = batch["image"]

        # alignment (optional)
        batch = self._apply_alignment(batch)

        rendering_out = self.rendering({dataset_key: batch})
        model_out = rendering_out["model_out"][dataset_key]
        model_image = model_out["image"]
        model_image, aux = self._apply_image_modules(batch, model_image)

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
        if 'correction' in aux:
            for k, v in aux['correction'].items():
                result[f"correction.{k}"] = v
        if 'background' in aux:
            result["background"] = aux['background']

        # prune None values (keeps callbacks simpler)
        return {k: v for k, v in result.items() if v is not None}

    def _val_query_points(self, batch):
        query_points = batch["query_points"]
        model_out = self.model(query_points)
        rho_pred = model_out["rho"]
        instrument_key = batch.get("instrument", None)
        if instrument_key in self.calibration_modules:
            rho_pred = self.calibration_modules[instrument_key](rho_pred)
        result = {
            "rho_pred": rho_pred,
            "v_pred": model_out["v"],
            "query_points": query_points,
        }
        if "density" in batch:
            result["density"] = batch["density"]
        if "density_cm3" in batch:
            result["density_cm3"] = batch["density_cm3"]
        if "velocity_radial" in batch:
            result["velocity_radial"] = batch["velocity_radial"]
        if "velocity_radial_kms" in batch:
            result["velocity_radial_kms"] = batch["velocity_radial_kms"]
        if "has_density" in batch:
            result["has_density"] = batch["has_density"]
        if "has_velocity" in batch:
            result["has_velocity"] = batch["has_velocity"]
        if "r_hat" in batch:
            result["r_hat"] = batch["r_hat"]
            result["velocity_radial_pred"] = (model_out["v"] * batch["r_hat"]).sum(dim=-1, keepdim=True)
        for key in ("radius_rsun", "time_norm", "time_days", "time_unix", "density_source"):
            if key in batch:
                result[key] = batch[key]
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

        batch = self._apply_alignment(batch)
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
                self.log(f'lambda_{k}', v['value'].detach().item(), sync_dist=True)
            if v['type'] == 'exponential_growth':
                new_value = v['value'] * v['gamma']
                if new_value >= v['end']:
                    new_value = v['end']
                v['value'] = new_value
                self.log(f'lambda_{k}', v['value'].detach().item(), sync_dist=True)
            if v['type'] == 'step':
                if self.global_step > int(v['steps'].item()):
                    v['value'].copy_(v['end'])
                self.log(f'lambda_{k}', v['value'].detach().item(), sync_dist=True)
            if v['type'] == 'constant':
                pass  # no change required, no logging
        # log instrument scaling
        scaling = {f'instrument_calibration.{k}': torch.exp(m.calibration).detach().item()
                   for k, m in self.calibration_modules.items()}
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'current_alpha'):
            scaling['dynamic_alpha'] = self.model.model.current_alpha.detach().item()
        self.log_dict(scaling, sync_dist=True)
        # call super method
        super().on_train_batch_end(*args, **kwargs)


def save_thomson_sunerf(sunerf: ThomsonSuNeRFModule, data_module: BaseDataModule, save_path,
                        msb_norm=None, msb=None, sigma_ne=None):
    output_path = '/'.join(save_path.split('/')[0:-1])
    os.makedirs(output_path, exist_ok=True)
    state = {
        # sunerf  rendering module
        'rendering': sunerf.rendering,
        'correction_modules': copy.deepcopy(sunerf.correction_modules).cpu(),
        'calibration_modules': copy.deepcopy(sunerf.calibration_modules).cpu(),
        # data infor
        'data_config': data_module.config,
        # data scaling
        'Rs_per_ds': data_module.Rs_per_ds,
        'seconds_per_dt': data_module.seconds_per_dt,
        'ref_date': data_module.ref_date,
        'thomson_normalization': {
            'msb_norm': msb_norm,
            'msb': msb,
            'sigma_ne': sigma_ne,
            'drho_cm3': data_module.drho_cm3,
        }
    }
    torch.save(state, save_path)
