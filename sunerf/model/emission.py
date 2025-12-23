import torch
from torch import nn

from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.rendering.emission import EmissionRadiativeTransfer
from sunerf.train.scaling import ImageAsinhScaling


class EmissionSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt, image_scaling_config,
                 lambda_image=1.0, lambda_regularization=1.0,
                 sampling_config=None, hierarchical_sampling_config=None,
                 model_config=None, **kwargs):

        self.lambda_image = lambda_image
        self.lambda_regularization = lambda_regularization

        # setup rendering
        rendering = EmissionRadiativeTransfer(Rs_per_ds=Rs_per_ds,
                                              sampling_config=sampling_config,
                                              hierarchical_sampling_config=hierarchical_sampling_config,
                                              model_config=model_config)

        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        self.image_scaling = ImageAsinhScaling(**image_scaling_config)
        self.mse_loss = nn.MSELoss()

    def training_step(self, batch, batch_nb):
        rays, time, target_image = batch['tracing']['rays'], batch['tracing']['time'], batch['tracing']['image']
        rays_o, rays_d = rays[:, 0], rays[:, 1]
        # Run one iteration of TinyNeRF and get the rendered filtergrams.
        outputs = self.rendering(rays_o, rays_d, time)

        # Check for any numerical issues.
        for k, v in outputs.items():
            assert not torch.isnan(v).any(), f"! [Numerical Alert] {k} contains NaN."
            assert not torch.isinf(v).any(), f"! [Numerical Alert] {k} contains Inf."

        # backpropagation
        target_image = self.image_scaling(target_image)
        # optimize coarse model
        coarse_image = self.image_scaling(outputs['coarse_image'])
        coarse_loss = self.mse_loss(coarse_image, target_image)
        # optimize fine model
        fine_image = self.image_scaling(outputs['fine_image'])
        fine_loss = self.mse_loss(fine_image, target_image)

        # regularization_loss = outputs['regularization'].mean()  # suppress unconstrained regions
        loss = (self.lambda_image * (coarse_loss + fine_loss))
        #
        with torch.no_grad():
            psnr = -10. * torch.log10(fine_loss)

        # log results to WANDB
        self.log("loss", loss)
        self.log("train",
                 {'coarse': coarse_loss, 'fine': fine_loss,
                  # 'regularization': regularization_loss,
                  'psnr': psnr})

        return loss

    def validation_step(self, batch, batch_nb, **kwargs):
        dataloader_idx = kwargs['dataloader_idx'] if 'dataloader_idx' in kwargs else 0
        if dataloader_idx == 0:
            rays, time, target_image = batch['rays'], batch['time'], batch['image']
            rays_o, rays_d = rays[:, 0], rays[:, 1]

            outputs = self.rendering(rays_o, rays_d, time)

            distance = rays_o.pow(2).sum(-1).pow(0.5)
            return {'target_image': target_image,
                    'fine_image': outputs['fine_image'],
                    'coarse_image': outputs['coarse_image'],
                    'height_map': outputs['height_map'],
                    'absorption_map': outputs['absorption_map'],
                    'z_vals_stratified': outputs['z_vals_stratified'],
                    'z_vals_hierarchical': outputs['z_vals_hierarchical'],
                    'distance': distance}
