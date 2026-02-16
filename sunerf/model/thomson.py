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
from sunerf.train.correction import CorrectionModule, CalibrationModule, AlignmentModule
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
        for instrument_config in instruments:
            instrument_config = instrument_config.copy()
            instrument_key = instrument_config.pop('key')
            instrument_type = instrument_config.pop('type')
            correction = instrument_config.pop('correction', False)
            calibration = instrument_config.pop('calibration', False)
            alignment = instrument_config.pop('alignment', False)
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
        available_lambdas = ['image', 'ratio', 'continuity', 'radial', 'velocity', 'target_velocity',
                             'f_corona', 'transmission', 'calibration_gain', 'calibration_offset', 'calibration_scalar',
                             'pB_mul', 'tB_mul', 'pB_add', 'tB_add']
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
        self.mse_loss = nn.MSELoss()

        # solar wind
        velocity_min = (100.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_min = nn.Parameter(torch.tensor(velocity_min, dtype=torch.float32), requires_grad=False)
        velocity_max = (800.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_max = nn.Parameter(torch.tensor(velocity_max, dtype=torch.float32), requires_grad=False)
        velocity_avg = (300.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_avg = nn.Parameter(torch.tensor(velocity_avg, dtype=torch.float32), requires_grad=False)

        # radial weighting
        self.min_radius_weight = nn.Parameter(torch.tensor(3.0 / Rs_per_ds, dtype=torch.float32), requires_grad=False)
        self.max_radius_weight = nn.Parameter(torch.tensor(10.0 / Rs_per_ds, dtype=torch.float32), requires_grad=False)

        print(f'Velocity min: {velocity_min}, max: {velocity_max}')
        drop_off_distance = (1 * u.AU).to_value(u.R_sun) / Rs_per_ds
        self.drop_off_distance = nn.Parameter(torch.tensor(drop_off_distance, dtype=torch.float32), requires_grad=False)

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

            model_image = model_out[k]['image']
            target_image = dataset_batch[k]['image']

            if instrument_key in self.correction_modules:
                model_image, correction = self.correction_modules[instrument_key](model_image, image_coords, hpc_coords,
                                                                                  time)
                # compute correction losses
                correction_losses.append(self.get_correction_loss(correction))
            if instrument_key in self.calibration_modules:
                model_image = self.calibration_modules[instrument_key](model_image)

            pB_nan_mask = ~torch.isnan(target_image[..., 1])

            # compute polarization ratios
            ratio_target_image = target_image[pB_nan_mask, 1] / (target_image[pB_nan_mask, 0] + 1e-8)
            ratio_model_image = model_image[pB_nan_mask, 1] / (model_image[pB_nan_mask, 0] + 1e-8)

            # scale images
            image_scaling = self.scaling_modules[instrument_key]
            scaled_model_image = image_scaling(model_image)
            scaled_target_image = image_scaling(target_image)

            # backpropagation
            # optimize model
            tB_image_diff = (scaled_model_image[..., 0] - scaled_target_image[..., 0]).pow(2)
            pB_image_diff = (scaled_model_image[pB_nan_mask, 1] - scaled_target_image[pB_nan_mask, 1]).pow(2)
            ratio_diff = (ratio_model_image - ratio_target_image).pow(2)

            instrument_tB_image_diff.append(tB_image_diff)
            instrument_pB_image_diff.append(pB_image_diff)
            instrument_ratio_diff.append(ratio_diff)

        tB_image_loss = torch.cat(instrument_tB_image_diff).mean()
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
            v_abs = torch.norm(v, dim=-1)
            min_v = torch.clip(v_abs - self.velocity_min, max=0).pow(2)
            max_v = torch.clip(v_abs - self.velocity_max, min=0).pow(2)
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

            # target velocity regularization
            target_velocity = query_points[:, :3] / (torch.norm(query_points[:, :3], dim=-1, keepdim=True) + 1e-7)
            target_velocity = target_velocity * self.velocity_avg
            target_loss = (v - target_velocity).pow(2).sum(-1)
            target_loss = (target_loss * radial_weight).sum() / (radial_weight.sum() + 1e-7)
            log_values['target_velocity'] = target_loss
            loss += self.lambdas['target_velocity']['value'] * target_loss

            assert torch.isnan(continuity_loss).sum() == 0, 'Invalid loss detected: continuity_loss'
            assert torch.isnan(velocity_loss).sum() == 0, 'Invalid loss detected: velocity_loss'
            assert torch.isnan(radial_loss).sum() == 0, 'Invalid loss detected: radial_loss'
            assert torch.isnan(target_loss).sum() == 0, 'Invalid loss detected: target_loss'

        assert torch.isnan(loss).sum() == 0, 'Invalid loss detected: loss'
        # log results to WANDB
        self.log("loss", loss)
        self.log_dict({f'train.{k}': v for k, v in log_values.items()})

        return loss

    def get_correction_loss(self, correction):
        correction_losses = {}
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

    def validation_step(self, batch, batch_nb, *args):
        dataloader_idx = args[0] if len(args) > 0 else 0
        dataset_key = self.validation_dataset_mapping[dataloader_idx]

        if 'instrument' in batch:
            instrument_key = batch['instrument']
            image_coords = batch['image_coords']
            hpc_coords = batch['hpc_coords']
            image = batch['image']
            time = batch['time']

            if instrument_key in self.alignment_modules:
                b_rays = batch['rays']
                aligned_rays = self.alignment_modules[instrument_key](b_rays, time)
                batch['rays'] = aligned_rays

            rendering_out = self.rendering({dataset_key: batch})

            model_out = rendering_out['model_out']
            model_out = model_out[dataset_key]

            model_image = model_out['image']

            if instrument_key in self.correction_modules:
                model_image, corrections = self.correction_modules[instrument_key](model_image, image_coords,
                                                                                   hpc_coords, time)
            else:
                corrections = None
            if instrument_key in self.calibration_modules:
                model_image = self.calibration_modules[instrument_key](model_image)

            target_ratio = image[..., 1:2] / (image[..., 0:1] + 1e-8)
            model_ratio = model_image[..., 1:2] / (model_image[..., 0:1] + 1e-8)

            image_scaling = self.scaling_modules[instrument_key]
            target_image = image_scaling(image)
            model_image = image_scaling(model_image)

            result = {'target_image': target_image,
                      'model_image': model_image,
                      'model_ratio': model_ratio,
                      'target_ratio': target_ratio,
                      'density': model_out['density'],
                      'distance_from_sun': model_out['distance_from_sun'],
                      'distance_from_obs': model_out['distance_from_obs'],
                      'distance': model_out['distance']}
            if corrections is not None:
                for k, v in corrections.items():
                    result[f'correction.{k}'] = v
            return result
        elif 'rho' in batch:
            query_points = batch['query_points']
            spherical_coords = batch['spherical_coords']
            rho_true = batch['rho']
            model_out = self.model(query_points)

            return {'rho_true': rho_true, 'rho_pred': model_out['rho'], 'v_pred': model_out['v'],
                    'spherical_coords': spherical_coords, 'query_points': query_points}
        else:
            query_points = batch['query_points']
            model_out = self.model(query_points)
            return {'rho_pred': model_out['rho'], 'v_pred': model_out['v'], 'query_points': query_points}

    def on_train_batch_end(self, *args, **kwargs):
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
        self.log_dict(scaling, sync_dist=True)
        # call super method
        super().on_train_batch_end(*args, **kwargs)


def save_thomson_sunerf(sunerf: ThomsonSuNeRFModule, data_module: BaseDataModule, save_path):
    output_path = '/'.join(save_path.split('/')[0:-1])
    os.makedirs(output_path, exist_ok=True)
    state = {
        # sunerf  rendering module
        'rendering': sunerf.rendering,
        # data infor
        'data_config': data_module.config,
        # data scaling
        'Rs_per_ds': data_module.Rs_per_ds,
        'seconds_per_dt': data_module.seconds_per_dt,
        'ref_date': data_module.ref_date,
    }
    torch.save(state, save_path)
