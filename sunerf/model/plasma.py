import os

import numpy as np
import torch
from torch import nn

from sunerf.data.loader.base_loader import BaseDataModule
from sunerf.model.model import PlasmaModel
from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.util import jacobian
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.plasma import PlasmaRadiativeTransfer, init_absorption_model
from sunerf.train.scaling import ImageAsinhScaling, ImageLinearScaling, ImageLogScaling


class PlasmaSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt, instruments_config,
                 lambda_config=None, sampling_config=None, hierarchical_sampling_config=None, absorption_config=None,
                 model_config=None, shuffle_config=None, **kwargs):
        # Temperature range
        log_T_range = np.arange(4, 8.001, 0.01).astype(np.float32)
        self.log_T_range = log_T_range

        # absorption model
        absorption_config = absorption_config if absorption_config is not None else {'type': 'learned'}
        absorption_model = init_absorption_model(absorption_config)

        # setup rendering
        rendering_modules = {}
        scaling_modules = {}
        for instrument in instruments_config:
            instrument = instrument.copy()
            instrument_key = instrument.pop('key')
            instrument_type = instrument.pop('type')
            if instrument_type == 'plasma':
                rendering_modules[instrument_key] = PlasmaRadiativeTransfer(
                    temperature_response_config=instrument['temperature_response'],
                    log_T_range=log_T_range, absorption_model=absorption_model)
            else:
                raise ValueError(f"Unknown instrument type: {instrument_type}")
            # image scaling
            scaling_config = instrument.pop('scaling', {})
            scaling_type = scaling_config.pop('type', 'asinh')
            if scaling_type == 'asinh':
                scaling_modules[instrument_key] = ImageAsinhScaling(**scaling_config)
            elif scaling_type == 'linear':
                scaling_modules[instrument_key] = ImageLinearScaling(**scaling_config)
            elif scaling_type == 'log':
                scaling_modules[instrument_key] = ImageLogScaling(**scaling_config)
            else:
                raise ValueError(f"Unknown scaling type: {scaling_type}")

        model = PlasmaModel(log_T=log_T_range, **model_config)
        rendering = BasicRenderingModule(model=model,
                                         rendering_modules=rendering_modules,
                                         Rs_per_ds=Rs_per_ds,
                                         sampling_config=sampling_config,
                                         hierarchical_sampling_config=hierarchical_sampling_config,
                                         shuffle_config=shuffle_config)

        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        lambda_config = lambda_config if lambda_config is not None else {}
        self.lambda_image = lambda_config.get('image', 1.0)
        self.lambda_regularization = lambda_config.get('regularization', 1.0e-4)
        self.lambda_absorption = lambda_config.get('absorption', 1.0e-4)

        self.absorption_model = absorption_model
        self.image_scaling = nn.ModuleDict(scaling_modules)
        self.mse_loss = nn.MSELoss()
        self.temperature_response_normalization = {k: v.normalization for k, v in rendering_modules.items()}

    def training_step(self, batch, batch_nb):
        instrument_batch = {k: v for k, v in batch.items() if k != 'random'}
        instruments = [v['instrument'] for v in instrument_batch.values()]
        ds_keys = list(instrument_batch.keys())
        rendering_out = self.rendering(instrument_batch)['model_out']

        image_diff = []
        absorption_regularization = []
        density_regularization = []

        for inst_key, ds_key in zip(instruments, ds_keys):
            pred_image = self.image_scaling[inst_key](rendering_out[ds_key]['image'])
            target_image = self.image_scaling[inst_key](instrument_batch[ds_key]['image'])

            # Check for any numerical issues.
            assert not torch.isnan(pred_image).any(), f"! [Numerical Alert] predicted image contains NaN."

            image_diff.append((pred_image - target_image).pow(2).sum(-1))

            absorption_regularization.append(rendering_out[ds_key]['mean_absorption'].mean())

            density = rendering_out[ds_key]['em'] * torch.clip(rendering_out[ds_key]['distance'] - 1.2, min=0) ** 2
            density_regularization.append(density.mean())

        image_loss = torch.cat(image_diff).mean()
        absorption_regularization = torch.stack(absorption_regularization).mean()
        density_regularization = torch.stack(density_regularization).mean()

        if 'random' in batch:
            query_points = batch['random']['coords']
            query_points.requires_grad = True

            rendering_out = self.rendering.model(query_points)
            total_ne = rendering_out['total_ne']
            mean_log_T = rendering_out['mean_log_T']
            regularization = self.compute_static_regularization(total_ne, mean_log_T, query_points)
        else:
            regularization = density_regularization

        loss = (self.lambda_image * image_loss +
                self.lambda_regularization * regularization +
                self.lambda_absorption * absorption_regularization)
        #
        with torch.no_grad():
            psnr = -10. * torch.log10(image_loss)

        # log results to WANDB
        self.log("loss", loss)
        self.log("train",
                 {'image': image_loss, 'psnr': psnr,
                  'density_regularization': regularization, 'absorption_regularization': absorption_regularization})

        return loss

    def compute_continuity(self, total_ne, velocity, query_points):
        in_tensor = torch.cat([total_ne, velocity], dim=-1)

        jac_matrix = jacobian(in_tensor, query_points)

        dRho_dx = jac_matrix[:, 0, 0]
        dVx_dx = jac_matrix[:, 1, 0]
        dRho_dy = jac_matrix[:, 0, 1]
        dVy_dy = jac_matrix[:, 2, 1]
        dRho_dz = jac_matrix[:, 0, 2]
        dVz_dz = jac_matrix[:, 3, 2]
        dRho_dt = jac_matrix[:, 0, 3]

        div_v = (dVx_dx + dVy_dy + dVz_dz)
        grad_rho = torch.stack([dRho_dx, dRho_dy, dRho_dz], -1)
        v_dot_grad_rho = (velocity * grad_rho).sum(-1)
        continuity_eq = dRho_dt + total_ne * div_v + v_dot_grad_rho
        regularization = continuity_eq.pow(2).mean()
        return regularization

    def compute_static_regularization(self, total_ne, mean_log_T, query_points):
        in_tensor = torch.cat([total_ne, mean_log_T], dim=-1)

        jac_matrix = jacobian(in_tensor, query_points)

        dRho_dx = jac_matrix[:, 0, 0]
        dRho_dy = jac_matrix[:, 0, 1]
        dRho_dz = jac_matrix[:, 0, 2]
        dRho_dt = jac_matrix[:, 0, 3]
        dlogT_dx = jac_matrix[:, 1, 0]
        dlogT_dy = jac_matrix[:, 1, 1]
        dlogT_dz = jac_matrix[:, 1, 2]
        dlogT_dt = jac_matrix[:, 1, 3]

        radius = torch.norm(query_points[..., :3], dim=-1)
        radius_weight = torch.clip(radius - 1.1, min=0) ** 2
        regularization = (dRho_dt * radius_weight + dlogT_dt * radius_weight)
        regularization = regularization.pow(2).mean()
        return regularization

    def validation_step(self, batch, batch_nb, *args):
        dataloader_idx = args[0] if len(args) > 0 else 0
        valid_ds_id = self.validation_dataset_mapping[dataloader_idx]
        if valid_ds_id == 'absorption':
            log_T, log_ne = batch['log_T'], batch['log_ne']
            absorption_input = torch.cat([log_ne, log_T], dim=-1)
            abs_out = self.absorption_model(absorption_input)
            return {**abs_out, 'log_T': log_T, 'log_ne': log_ne}
        else:
            ds_key = self.validation_dataset_mapping[dataloader_idx]
            image = batch['image']
            instrument_key = batch['instrument']

            rendering_out = self.rendering({ds_key: batch})
            model_out = rendering_out['model_out']

            image_scaling = self.image_scaling[instrument_key]

            image = torch.nan_to_num(image, nan=0.0)

            target_image = image_scaling(image)
            pred_image = image_scaling(model_out[ds_key]['image'])

            # set nans to zero
            target_image = torch.nan_to_num(target_image, nan=0.0)
            pred_image = torch.nan_to_num(pred_image, nan=0.0)

            return {'target_image': target_image,
                    'pred_image': pred_image,
                    'mean_T': model_out[ds_key]['mean_T'], 'total_ne': model_out[ds_key]['total_ne'],
                    'height_map': model_out[ds_key]['height_map'],
                    'mean_absorption': model_out[ds_key]['mean_absorption'],
                    'z_vals_stratified': rendering_out['z_vals_stratified'],
                    'z_vals_hierarchical': rendering_out['z_vals'],
                    'distance': model_out[ds_key]['distance']}

    def validation_epoch_end(self, *args, **kwargs):
        scaling = {k: float(m.instrument_scaling.detach().cpu().numpy())
                   for k, m in self.rendering.rendering_modules.items()}
        self.log(f'instrument_scaling', scaling)
        super().validation_epoch_end(*args, **kwargs)


def save_plasma_sunerf(sunerf: PlasmaSuNeRFModule, data_module: BaseDataModule, save_path):
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
        'temperature_response_normalization': sunerf.temperature_response_normalization,
        'log_T_range': sunerf.log_T_range,
    }
    torch.save(state, save_path)
