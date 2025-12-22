import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from sunerf.model.model import ImageToLatentCNN, CoordinateToLatentModel
from sunerf.model.sunerf import BaseModule
from sunerf.rendering.conditioned_emission import ConditionedRadiativeTransfer
from sunerf.train.scaling import ImageAsinhScaling


class ConditionedSuNeRFModule(BaseModule):
    def __init__(self, Rs_per_ds, image_scaling_config, in_channels,
                 lambda_config=None,
                 sampling_config=None,
                 model_config=None, lr_config=None, use_absorption=True,
                 **kwargs):
        super().__init__(**kwargs)
        lambda_config = {'image': 1.0} if lambda_config is None else lambda_config
        self.lambda_image = lambda_config.get('image', 1.0)

        # setup rendering
        z_dim = model_config.pop('latent_z', 1024)
        self.rendering = ConditionedRadiativeTransfer(z_dim=z_dim + 32,
                                                      Rs_per_ds=Rs_per_ds,
                                                      sampling_config=sampling_config,
                                                      model_config=model_config, use_absorption=use_absorption)
        self.encoder = ImageToLatentCNN(in_channels, z_dim=z_dim)
        self.coordinate_encoder = CoordinateToLatentModel(z_dim=32)

        self.image_scaling = ImageAsinhScaling(**image_scaling_config)
        self.mse_loss = nn.MSELoss()

        self.lr_config = {'start': 1e-4, 'end': 1e-5, 'iterations': 1e6} if lr_config is None else lr_config

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_config['start'])
        self.scheduler = ExponentialLR(self.optimizer, gamma=(self.lr_config['end'] / self.lr_config['start']) ** (
                1 / self.lr_config['iterations']))
        return [self.optimizer], [self.scheduler]

    def on_train_batch_end(self, *args, **kwargs):
        # update learning rate and log
        if self.scheduler.get_last_lr()[0] > 5e-5:
            self.scheduler.step()
        self.log('Learning Rate', self.scheduler.get_last_lr()[0])

    def training_step(self, batch, batch_nb):
        rays, target_image, input_image = batch['rays'], batch['target_image'], batch['input_image']
        rays_o, rays_d = rays[..., 0, :], rays[..., 1, :] # [batch, n_rays, 3]
        latitude, longitude = batch['latitude'], batch['longitude']
        observer_coords = torch.stack([torch.sin(latitude), torch.cos(latitude),
                                       torch.sin(longitude), torch.cos(longitude)], dim=-1)

        # scale target image
        input_image = self.image_scaling(input_image) # [batch, C, H, W]

        # get feature vector from encoding network
        latent_img_z = self.encoder(input_image)
        latent_coord_z = self.coordinate_encoder(observer_coords)  # [batch, 4]
        latent_z = torch.cat([latent_img_z, latent_coord_z], dim=-1)  # [batch, z_dim + 32]

        # flatten and repeat latent vector for all rays in the batch
        latent_z = latent_z[:, None, :].repeat(1, rays_o.shape[1], 1)  # [batch, n_rays, z_dim]

        flat_latent_z = latent_z.view(-1, latent_z.shape[-1])  # [batch * n_rays, z_dim]
        flat_rays_o = rays_o.view(-1, rays_o.shape[-1])  # [batch * n_rays, 3]
        flat_rays_d = rays_d.view(-1, rays_d.shape[-1])  # [batch * n_rays, 3]

        outputs = self.rendering(flat_rays_o, flat_rays_d, flat_latent_z)

        # prep predicted image
        predicted_image = outputs['model_out']['image'].view(target_image.shape)
        predicted_image = self.image_scaling(predicted_image)

        # prep target image
        target_image = self.image_scaling(target_image)

        # optimize model
        image_loss = self.mse_loss(predicted_image, target_image)

        loss = (self.lambda_image * image_loss)
        #
        with torch.no_grad():
            psnr = -10. * torch.log10(image_loss)

        # log results to WANDB
        self.log_dict({"loss": loss, 'train.image': image_loss, 'train.psnr': psnr})

        return loss

    def validation_step(self, batch, batch_nb, **kwargs):
        dataloader_idx = kwargs['dataloader_idx'] if 'dataloader_idx' in kwargs else 0
        if dataloader_idx == 0:
            rays, target_image, input_image = batch['rays'], batch['target_image'], batch['input_image']
            rays_o, rays_d = rays[..., 0, :], rays[..., 1, :] # [batch, n_rays, 3]
            latitude, longitude = batch['latitude'], batch['longitude']
            observer_coords = torch.stack([torch.sin(latitude), torch.cos(latitude),
                                           torch.sin(longitude), torch.cos(longitude)], dim=-1)

            # scale target image
            input_image = self.image_scaling(input_image)  # [batch, C, H, W]

            # get feature vector from encoding network
            latent_img_z = self.encoder(input_image)
            latent_coord_z = self.coordinate_encoder(observer_coords)
            latent_z = torch.cat([latent_img_z, latent_coord_z], dim=-1)  # [batch, z_dim + 32]

            # flatten and repeat latent vector for all rays in the batch
            latent_z = latent_z[:, None, :].repeat(1, rays_o.shape[1], 1)  # [batch, n_rays, z_dim]

            flat_latent_z = latent_z.view(-1, latent_z.shape[-1])  # [batch * n_rays, z_dim]
            flat_rays_o = rays_o.view(-1, rays_o.shape[-1])  # [batch * n_rays, 3]
            flat_rays_d = rays_d.view(-1, rays_d.shape[-1])  # [batch * n_rays, 3]

            outputs = self.rendering(flat_rays_o, flat_rays_d, flat_latent_z)

            distance = rays_o.pow(2).sum(-1).pow(0.5)
            target_shape = target_image.shape[:2]

            predicted_image = outputs['model_out']['image'].view(*target_shape, -1)
            predicted_image = self.image_scaling(predicted_image)
            target_image = self.image_scaling(target_image)

            out = {'target_image': target_image,
                    'predicted_image': predicted_image,
                    'height_map': outputs['model_out']['height_map'].view(*target_shape, -1),
                    'absorption_map': outputs['model_out']['absorption_map'].view(*target_shape, -1),
                    'z_vals': outputs['z_vals'].view(*target_shape, -1),
                    'distance': distance}
            out = {k: v[0] for k, v in out.items()}  # remove batch dim (1,)
            return out
