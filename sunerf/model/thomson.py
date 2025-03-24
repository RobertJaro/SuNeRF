import torch
from astropy import units as u
from torch import nn

from sunerf.model.model import ThomsonModel
from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.util import jacobian
from sunerf.rendering.base_tracing import MultiResolutionRenderingModule
from sunerf.rendering.thomson import ThomsonScattering
from sunerf.train.scaling import ImageAsinhScaling, ImageLinearScaling, ImageLogScaling


class ThomsonSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt,
                 instruments,
                 lambda_image=1.0, lambda_ratio=1.0,
                 lambda_continuity=1e-3, lambda_radial=1e-2, lambda_velocity=1e-3,
                 sampling_config=None, hierarchical_sampling_config=None,
                 model_config=None, **kwargs):
        # setup rendering
        sampling_config = sampling_config if sampling_config is not None else {}
        hierarchical_sampling_config = hierarchical_sampling_config if hierarchical_sampling_config is not None else {}

        rendering_modules = {}
        scaling_modules = {}
        for instrument_config in instruments:
            instrument_config = instrument_config.copy()
            instrument_key = instrument_config.pop('key')
            instrument_type = instrument_config.pop('type')
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

        model_config = {} if model_config is None else model_config
        coarse_model = ThomsonModel(**model_config)
        fine_model = ThomsonModel(**model_config)
        rendering = MultiResolutionRenderingModule(coarse_model=coarse_model, fine_model=fine_model,
                                                   rendering_modules=rendering_modules,
                                                   Rs_per_ds=Rs_per_ds,
                                                   sampling_config=sampling_config,
                                                   hierarchical_sampling_config=hierarchical_sampling_config)

        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        self.rendering_modules = rendering_modules
        self.coarse_model = coarse_model
        self.fine_model = fine_model

        self.lambda_image = lambda_image
        self.lambda_ratio = lambda_ratio
        self.lambda_continuity = lambda_continuity
        self.lambda_radial = lambda_radial
        self.lambda_velocity = lambda_velocity

        self.scaling_modules = nn.ModuleDict(scaling_modules)
        self.mse_loss = nn.MSELoss()

        # solar wind
        velocity_min = (200.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # Mm/s --> ds/dt
        self.velocity_min = nn.Parameter(torch.tensor(velocity_min, dtype=torch.float32), requires_grad=False)
        velocity_max = (800.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # Mm/s --> ds/dt
        self.velocity_max = nn.Parameter(torch.tensor(velocity_max, dtype=torch.float32), requires_grad=False)

        drop_off_distance = (1 * u.AU).to_value(u.R_sun) / Rs_per_ds
        self.drop_off_distance = nn.Parameter(torch.tensor(drop_off_distance, dtype=torch.float32), requires_grad=False)

    def training_step(self, batch, batch_nb):
        dataset_batch = {k: v for k, v in batch.items() if k != 'random'}
        rendering_out = self.rendering(dataset_batch)

        fine_out = rendering_out['fine_out']
        coarse_out = rendering_out['coarse_out']

        coarse_diff = []
        caorse_ratio_diff = []
        fine_diff = []
        fine_ratio_diff = []

        for k in dataset_batch.keys():
            instrument_key = batch[k]['instrument']
            image_scaling = self.scaling_modules[instrument_key]

            coarse_image = coarse_out[k]['image']
            fine_image = fine_out[k]['image']
            target_image = dataset_batch[k]['image']

            # compute polarization ratios
            ratio_target_image = target_image[..., 1] / (target_image[..., 0] + 1e-6)
            ratio_coarse_image = coarse_image[..., 1] / (coarse_image[..., 0] + 1e-6)
            ratio_fine_image = fine_image[..., 1] / (fine_image[..., 0] + 1e-6)

            # scale images
            scaled_coarse_image = image_scaling(coarse_image)
            scaled_fine_image = image_scaling(fine_image)
            scaled_target_image = image_scaling(target_image)

            # Check for any numerical issues.
            assert not torch.isnan(scaled_coarse_image).any(), f"! [Numerical Alert] target_image contains NaN."
            assert not torch.isnan(scaled_fine_image).any(), f"! [Numerical Alert] target_image contains NaN."

            # backpropagation

            # optimize coarse model
            image_diff = (scaled_coarse_image - scaled_target_image).pow(2).sum(-1)
            ratio_diff = (ratio_coarse_image - ratio_target_image).pow(2)
            coarse_diff.append(image_diff)
            caorse_ratio_diff.append(ratio_diff)

            # optimize fine model
            image_diff = (scaled_fine_image - scaled_target_image).pow(2).sum(-1)
            ratio_diff = (ratio_fine_image - ratio_target_image).pow(2)
            fine_diff.append(image_diff)
            fine_ratio_diff.append(ratio_diff)

        coarse_loss = torch.cat(coarse_diff).mean()
        fine_loss = torch.cat(fine_diff).mean()
        coarse_ratio_loss = torch.cat(caorse_ratio_diff).mean()
        fine_ratio_loss = torch.cat(fine_ratio_diff).mean()
        # density_regularization = torch.stack(density_regularization).mean()

        loss = self.lambda_image * (coarse_loss + fine_loss) + self.lambda_ratio * (coarse_ratio_loss + fine_ratio_loss)

        with torch.no_grad():
            psnr = -10. * torch.log10(fine_loss)

        log_values = {'coarse': coarse_loss, 'fine': fine_loss, 'psnr': psnr,
                      'coarse_ratio': coarse_ratio_loss, 'fine_ratio': fine_ratio_loss}

        if 'random' in batch:
            query_points = batch['random']['coords']
            query_points.requires_grad = True

            fine_out = self.fine_model(query_points)

            rho = fine_out['rho']
            v = fine_out['v']
            fine_continuity_loss = self.compute_continuity_loss(rho, v, query_points)

            coarse_out = self.coarse_model(query_points)

            rho = coarse_out['rho']
            v = coarse_out['v']
            coarse_continuity_loss = self.compute_continuity_loss(rho, v, query_points)

            continuity_loss = fine_continuity_loss + coarse_continuity_loss
            loss += self.lambda_continuity * continuity_loss
            log_values['continuity'] = continuity_loss

            # velocity regularization
            v_abs = torch.norm(fine_out['v'], dim=-1)
            min_v = torch.clip(v_abs - self.velocity_min, max=0).pow(2).mean()
            max_v = torch.clip(v_abs - self.velocity_max, min=0).pow(2).mean()
            velocity_regularization_fine = min_v + max_v

            v_abs = torch.norm(coarse_out['v'], dim=-1)
            min_v = torch.clip(v_abs - self.velocity_min, max=0).pow(2).mean()
            max_v = torch.clip(v_abs - self.velocity_max, min=0).pow(2).mean()
            velocity_regularization_coarse = min_v + max_v

            velocity_loss = velocity_regularization_fine + velocity_regularization_coarse
            log_values['velocity_loss'] = velocity_loss
            loss += self.lambda_velocity * velocity_loss

            # radial regularization
            normalization = torch.norm(query_points[:, :3], dim=-1) * torch.norm(fine_out['v'], dim=-1)
            radial_loss = 1 - (fine_out['v'] * query_points[:, :3]).sum(-1) / normalization
            radial_loss_fine = radial_loss.pow(2).mean()

            normalization = torch.norm(query_points[:, :3], dim=-1) * torch.norm(coarse_out['v'], dim=-1)
            radial_loss = 1 - (coarse_out['v'] * query_points[:, :3]).sum(-1) / normalization
            radial_loss_coarse = radial_loss.pow(2).mean()

            radial_loss = radial_loss_fine + radial_loss_coarse
            log_values['radial_loss'] = radial_loss
            loss += self.lambda_radial * radial_loss

        # log results to WANDB
        self.log("loss", loss)
        self.log("train", log_values)

        return loss

    def compute_continuity_loss(self, rho, v, query_points):
        in_tensor = torch.cat([rho, v], dim=-1)

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
        v_dot_grad_rho = (v * grad_rho).sum(-1)
        continuity_eq = dRho_dt + rho * div_v + v_dot_grad_rho
        regularization = continuity_eq.pow(2).mean()
        return regularization

    def validation_step(self, batch, batch_nb, *args):
        dataloader_idx = args[0] if len(args) > 0 else 0
        dataset_key = self.validation_dataset_mapping[dataloader_idx]

        if 'instrument' in batch:
            instrument_key = batch['instrument']

            image = batch['image']

            rendering_out = self.rendering({dataset_key: batch})

            fine_out = rendering_out['fine_out']
            coarse_out = rendering_out['coarse_out']

            fine_out = fine_out[dataset_key]
            coarse_out = coarse_out[dataset_key]

            fine_image = fine_out['image']
            coarse_image = coarse_out['image']

            image_scaling = self.scaling_modules[instrument_key]

            ratio_target_image = image[..., 1:2] / (image[..., 0:1] + 1e-6)
            ratio_fine_image = fine_image[..., 1:2] / (fine_image[..., 0:1] + 1e-6)
            ratio_coarse_image = coarse_image[..., 1:2] / (coarse_image[..., 0:1] + 1e-6)

            target_image = image_scaling(image)
            fine_image = image_scaling(fine_image)
            coarse_image = image_scaling(coarse_image)

            return {'target_image': target_image,
                    'fine_image': fine_image,
                    'coarse_image': coarse_image,
                    'ratio_target_image': ratio_target_image,
                    'ratio_fine_image': ratio_fine_image, 'ratio_coarse_image': ratio_coarse_image,
                    'density': fine_out['density'],
                    'distance_from_sun': fine_out['distance_from_sun'],
                    'distance_from_obs': fine_out['distance_from_obs'],
                    'z_vals_stratified': coarse_out['z_vals'],
                    'z_vals_hierarchical': fine_out['z_vals'],
                    'distance': fine_out['distance']}
        else:
            query_points = batch['query_points']
            spherical_coords = batch['spherical_coords']
            rho_true = batch['rho']
            fine_out = self.fine_model(query_points)

            return {'rho_true': rho_true, 'rho_pred': fine_out['rho'], 'v_pred': fine_out['v'],
                    'spherical_coords': spherical_coords}


    def validation_epoch_end(self, *args, **kwargs):
        scaling = {k: float(m.scaling.detach().cpu().numpy())
                   for k, m in self.rendering_modules.items()}
        self.log(f'instrument_scaling', scaling)
        super().validation_epoch_end(*args, **kwargs)
