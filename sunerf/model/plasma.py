import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.util import jacobian
from sunerf.rendering.plasma import PlasmaRadiativeTransfer
from sunerf.train.scaling import ImageAsinhScaling


class PlasmaSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt,
                 image_scaling_config, temperature_response_config,
                 lambda_image=1.0, lambda_regularization=1.0e-4, lambda_absorption=1.0e-4,
                 sampling_config=None, hierarchical_sampling_config=None,
                 model_config=None, shuffle_config=None, **kwargs):
        # setup rendering
        rendering = PlasmaRadiativeTransfer(
            temperature_response_config=temperature_response_config, Rs_per_ds=Rs_per_ds,
            sampling_config=sampling_config,
            hierarchical_sampling_config=hierarchical_sampling_config,
            model_config=model_config, shuffle_config=shuffle_config, **kwargs)
        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)
        self.lambda_image = lambda_image
        self.lambda_regularization = lambda_regularization
        self.lambda_absorption = lambda_absorption

        image_scaling = {k: ImageAsinhScaling(**c) for k, c in image_scaling_config.items()}
        self.image_scaling = nn.ModuleDict(image_scaling)
        self.mse_loss = nn.MSELoss()

    def configure_optimizers(self):
        params = list(self.rendering.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr_config['start'])
        self.scheduler = ExponentialLR(self.optimizer, gamma=(self.lr_config['end'] / self.lr_config['start']) ** (
                1 / self.lr_config['iterations']))
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        instrument_batch = {k: v for k, v in batch.items() if k != 'random'}
        instruments = instrument_batch.keys()
        rendering_out = self.rendering(instrument_batch)
        fine_out, coarse_out = rendering_out['fine_out'], rendering_out['coarse_out']

        for k in instruments:
            assert torch.isnan(
                fine_out[k]['query_points']).any() == False, f"! [Numerical Alert] query_points contains NaN. In {k}"

        coarse_diff = []
        fine_diff = []
        absorption_regularization = []
        density_regularization = []

        for k in instruments:
            image_scaling = self.image_scaling[k]
            coarse_image = image_scaling(coarse_out[k]['image'])
            fine_image = image_scaling(fine_out[k]['image'])
            target_image = image_scaling(instrument_batch[k]['image'])

            # Check for any numerical issues.
            assert not torch.isnan(coarse_image).any(), f"! [Numerical Alert] target_image contains NaN."
            assert not torch.isnan(fine_image).any(), f"! [Numerical Alert] target_image contains NaN."

            # backpropagation
            # optimize coarse model
            # coarse_loss = (coarse_image - target_image) / target_image.mean(0, keepdim=True)
            # coarse_loss = coarse_loss.pow(2).mean()
            coarse_diff.append((coarse_image - target_image).pow(2).sum(-1))
            # optimize fine model
            # fine_loss = (fine_image - target_image) / target_image.mean(0, keepdim=True)
            # fine_loss = fine_loss.pow(2).mean()
            fine_diff.append((fine_image - target_image).pow(2).sum(-1))

            absorption_regularization.append(
                coarse_out[k]['mean_absorption'].mean() + fine_out[k]['mean_absorption'].mean())

            coarse_density = coarse_out[k]['em'] * torch.clip(coarse_out[k]['distance'] - 1.2, min=0) ** 2
            fine_density = fine_out[k]['em'] * torch.clip(fine_out[k]['distance'] - 1.2, min=0) ** 2
            density_regularization.append(coarse_density.mean() + fine_density.mean())

        coarse_loss = torch.cat(coarse_diff).mean()
        fine_loss = torch.cat(fine_diff).mean()
        absorption_regularization = torch.stack(absorption_regularization).mean()
        density_regularization = torch.stack(density_regularization).mean()

        if 'random' in batch:
            query_points = batch['random']['coords']
            query_points.requires_grad = True

            fine_out = self.rendering.fine_model(query_points)
            total_ne = fine_out['total_ne']
            mean_log_T = fine_out['mean_log_T']
            # velocity = fine_out['velocity']
            fine_regularization = self.compute_static_regularization(total_ne, mean_log_T, query_points)

            coarse_out = self.rendering.coarse_model(query_points)
            total_ne = coarse_out['total_ne']
            mean_log_T = coarse_out['mean_log_T']
            # velocity = coarse_out['velocity']
            coarse_regularization = self.compute_static_regularization(total_ne, mean_log_T, query_points)

            regularization = fine_regularization + coarse_regularization
        else:
            regularization = density_regularization

        loss = (self.lambda_image * (coarse_loss + fine_loss) +
                self.lambda_regularization * regularization +
                self.lambda_absorption * absorption_regularization)
        #
        with torch.no_grad():
            psnr = -10. * torch.log10(fine_loss)

        # log results to WANDB
        self.log("loss", loss)
        self.log("train",
                 {'coarse': coarse_loss, 'fine': fine_loss, 'psnr': psnr,
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
            abs_out = self.rendering.absorption_model(absorption_input)
            return {**abs_out, 'log_T': log_T, 'log_ne': log_ne}
        else:
            instrument_key = self.validation_dataset_mapping[dataloader_idx]
            image = batch['image']

            fine_out, coarse_out = self.rendering({instrument_key: batch})

            image_scaling = self.image_scaling[instrument_key]

            image = torch.nan_to_num(image, nan=0.0)

            target_image = image_scaling(image)
            fine_image = image_scaling(fine_out[instrument_key]['image'])
            coarse_image = image_scaling(coarse_out[instrument_key]['image'])

            # set nans to zero
            target_image = torch.nan_to_num(target_image, nan=0.0)
            fine_image = torch.nan_to_num(fine_image, nan=0.0)
            coarse_image = torch.nan_to_num(coarse_image, nan=0.0)

            return {'target_image': target_image,
                    'fine_image': fine_image,
                    'coarse_image': coarse_image,
                    'mean_T': fine_out[instrument_key]['mean_T'], 'total_ne': fine_out[instrument_key]['total_ne'],
                    'height_map': fine_out[instrument_key]['height_map'],
                    'mean_absorption': fine_out[instrument_key]['mean_absorption'],
                    'z_vals_stratified': coarse_out[instrument_key]['z_vals'],
                    'z_vals_hierarchical': fine_out[instrument_key]['z_vals'],
                    'distance': fine_out[instrument_key]['distance']}

    def validation_epoch_end(self, *args, **kwargs):
        scaling = {k: float(self.rendering.instrument_scaling[idx].detach().cpu().numpy())
                   for k, idx in self.rendering.temperature_response_mapping.items()}
        self.log(f'instrument_scaling', scaling)
        super().validation_epoch_end(*args, **kwargs)
