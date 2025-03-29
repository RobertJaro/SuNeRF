import torch
from astropy import units as u
from torch import nn

from sunerf.model.model import RhoModel
from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.util import jacobian
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.thomson import ThomsonScattering
from sunerf.train.scaling import ImageAsinhScaling, ImageLinearScaling, ImageLogScaling


class ThomsonSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt,
                 instruments,
                 lambda_image=1.0, lambda_ratio=1.0,
                 lambda_continuity=1e-3, lambda_radial=1e-2, lambda_velocity=1e-3,
                 sampling_config=None,
                 model_config=None, **kwargs):
        # setup rendering
        sampling_config = sampling_config if sampling_config is not None else {}

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
        model = RhoModel(**model_config)
        rendering = BasicRenderingModule(model=model,
                                         rendering_modules=rendering_modules,
                                         Rs_per_ds=Rs_per_ds,
                                         sampling_config=sampling_config)

        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        self.rendering_modules = rendering_modules
        self.model = model

        self.lambda_image = lambda_image
        self.lambda_ratio = lambda_ratio
        self.lambda_continuity = lambda_continuity
        self.lambda_radial = lambda_radial
        self.lambda_velocity = lambda_velocity

        self.scaling_modules = nn.ModuleDict(scaling_modules)
        self.mse_loss = nn.MSELoss()

        # solar wind
        velocity_min = (200.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_min = nn.Parameter(torch.tensor(velocity_min, dtype=torch.float32), requires_grad=False)
        velocity_max = (800.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt  # km/s --> ds/dt
        self.velocity_max = nn.Parameter(torch.tensor(velocity_max, dtype=torch.float32), requires_grad=False)

        print(f'Velocity min: {velocity_min}, max: {velocity_max}')
        drop_off_distance = (1 * u.AU).to_value(u.R_sun) / Rs_per_ds
        self.drop_off_distance = nn.Parameter(torch.tensor(drop_off_distance, dtype=torch.float32), requires_grad=False)

    def training_step(self, batch, batch_nb):
        dataset_batch = {k: v for k, v in batch.items() if k != 'random'}
        rendering_out = self.rendering(dataset_batch)

        model_out = rendering_out['model_out']

        instrument_image_diff = []
        instrument_ratio_diff = []

        for k in dataset_batch.keys():
            instrument_key = batch[k]['instrument']
            image_scaling = self.scaling_modules[instrument_key]

            model_image = model_out[k]['image']
            target_image = dataset_batch[k]['image']

            # compute polarization ratios
            ratio_target_image = target_image[..., 1] / (target_image[..., 0] + 1e-6)
            ratio_model_image = model_image[..., 1] / (model_image[..., 0] + 1e-6)

            # scale images
            scaled_model_image = image_scaling(model_image)
            scaled_target_image = image_scaling(target_image)

            # backpropagation
            # optimize model
            image_diff = (scaled_model_image - scaled_target_image).pow(2).sum(-1)
            ratio_diff = (ratio_model_image - ratio_target_image).pow(2)
            instrument_image_diff.append(image_diff)
            instrument_ratio_diff.append(ratio_diff)

        image_loss = torch.cat(instrument_image_diff).mean()
        ratio_loss = torch.cat(instrument_ratio_diff).mean()
        # density_regularization = torch.stack(density_regularization).mean()

        loss = self.lambda_image * image_loss + self.lambda_ratio * ratio_loss

        with torch.no_grad():
            psnr = -10. * torch.log10(image_loss)

        log_values = {'image': image_loss, 'psnr': psnr,
                      'ratio': ratio_loss}

        if 'random' in batch:
            query_points = batch['random']['coords']
            query_points.requires_grad = True

            model_out = self.model(query_points)

            rho = model_out['rho']
            v = model_out['v']
            continuity_loss = self.compute_continuity_loss(rho, v, query_points)

            loss += self.lambda_continuity * continuity_loss
            log_values['continuity'] = continuity_loss

            # velocity regularization
            v_abs = torch.norm(v, dim=-1)
            min_v = torch.clip(v_abs - self.velocity_min, max=0).pow(2).mean()
            max_v = torch.clip(v_abs - self.velocity_max, min=0).pow(2).mean()
            velocity_loss = min_v + max_v
            log_values['velocity'] = velocity_loss
            loss += self.lambda_velocity * velocity_loss

            # radial regularization
            normalization = torch.norm(query_points[:, :3], dim=-1) * torch.norm(v, dim=-1) + 1e-7
            radial_loss = 1 - (v * query_points[:, :3]).sum(-1) / normalization
            radial_loss = radial_loss.pow(2).mean()
            log_values['radial'] = radial_loss
            loss += self.lambda_radial * radial_loss

            # match target velocity profile
            # target_velocity = query_points[:, :3] / torch.norm(query_points[:, :3], dim=-1, keepdim=True)
            # velocity_loss = (v - target_velocity).pow(2).sum(-1).mean()
            # log_values['velocity_loss'] = velocity_loss
            # loss += self.lambda_velocity * velocity_loss

        # log results to WANDB
        self.log("loss", loss)
        self.log("train", log_values)

        return loss

    # def compute_continuity_loss(self, rho, v, query_points):
    #     rho_jac_matrix = jacobian(rho, query_points)
    #     dRho_dx = rho_jac_matrix[:, 0, 0]
    #     dRho_dy = rho_jac_matrix[:, 0, 1]
    #     dRho_dz = rho_jac_matrix[:, 0, 2]
    #     dRho_dt = rho_jac_matrix[:, 0, 3]
    #
    #     rho_v = rho * v
    #     v_jac_matrix = jacobian(rho_v, query_points)
    #     dRhoVx_dx = v_jac_matrix[:, 0, 0]
    #     dRhoVy_dy = v_jac_matrix[:, 1, 1]
    #     dRhoVz_dz = v_jac_matrix[:, 2, 2]
    #
    #     div_rho_v = dRhoVx_dx + dRhoVy_dy + dRhoVz_dz
    #     continuity_eq = dRho_dt + div_rho_v
    #
    #     loss = continuity_eq.abs()
    #     radial_distance = torch.norm(query_points[:, :3], dim=-1)
    #     # compensate for the radial drop-off
    #     loss = loss * radial_distance ** 2
    #     # normalize density
    #     loss = loss / (rho * radial_distance ** 2).mean()
    #
    #     return loss.mean()

    def compute_continuity_loss(self, rho, v, query_points):
        out = torch.cat([rho, v], dim=-1)
        jac_matrix = jacobian(out, query_points)

        dRho_dx = jac_matrix[:, 0, 0]
        dVx_dx = jac_matrix[:, 1, 0]
        dVy_dx = jac_matrix[:, 2, 0]
        dVz_dx = jac_matrix[:, 3, 0]

        dRho_dy = jac_matrix[:, 0, 1]
        dVx_dy = jac_matrix[:, 1, 1]
        dVy_dy = jac_matrix[:, 2, 1]
        dVz_dy = jac_matrix[:, 3, 1]

        dRho_dz = jac_matrix[:, 0, 2]
        dVx_dz = jac_matrix[:, 1, 2]
        dVy_dz = jac_matrix[:, 2, 2]
        dVz_dz = jac_matrix[:, 3, 2]

        dRho_dt = jac_matrix[:, 0, 3]
        dVx_dt = jac_matrix[:, 1, 3]
        dVy_dt = jac_matrix[:, 2, 3]
        dVz_dt = jac_matrix[:, 3, 3]

        div_v = (dVx_dx + dVy_dy + dVz_dz)
        grad_rho = torch.stack([dRho_dx, dRho_dy, dRho_dz], -1)
        v_dot_grad_rho = (v * grad_rho).sum(-1)
        continuity_loss = dRho_dt + rho * div_v + v_dot_grad_rho
        radius = torch.norm(query_points[..., :3], dim=-1)
        continuity_loss = continuity_loss.pow(2)  # * radius ** 2
        continuity_loss = continuity_loss.mean() / rho.mean()
        return continuity_loss

    def validation_step(self, batch, batch_nb, *args):
        dataloader_idx = args[0] if len(args) > 0 else 0
        dataset_key = self.validation_dataset_mapping[dataloader_idx]

        if 'instrument' in batch:
            instrument_key = batch['instrument']

            image = batch['image']

            rendering_out = self.rendering({dataset_key: batch})

            model_out = rendering_out['model_out']
            model_out = model_out[dataset_key]

            model_image = model_out['image']

            target_ratio = image[..., 1:2] / (image[..., 0:1] + 1e-6)
            model_ratio = model_image[..., 1:2] / (model_image[..., 0:1] + 1e-6)

            image_scaling = self.scaling_modules[instrument_key]
            target_image = image_scaling(image)
            model_image = image_scaling(model_image)

            return {'target_image': target_image,
                    'model_image': model_image,
                    'model_ratio': model_ratio,
                    'target_ratio': target_ratio,
                    'density': model_out['density'],
                    'distance_from_sun': model_out['distance_from_sun'],
                    'distance_from_obs': model_out['distance_from_obs'],
                    'distance': model_out['distance']}
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

    def validation_epoch_end(self, *args, **kwargs):
        scaling = {k: float(m.scaling.detach().cpu().numpy())
                   for k, m in self.rendering_modules.items()}
        self.log(f'instrument_scaling', scaling)
        super().validation_epoch_end(*args, **kwargs)
