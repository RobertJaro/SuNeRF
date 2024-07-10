import os

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from sunerf.data.loader.base_loader import BaseDataModule
from sunerf.rendering.base_tracing import SuNeRFRendering
from sunerf.rendering.emission import EmissionRadiativeTransfer
from sunerf.train.scaling import ImageAsinhScaling


class BaseSuNeRFModule(LightningModule):

    def __init__(self, Rs_per_ds, seconds_per_dt, rendering: SuNeRFRendering,
                 validation_dataset_mapping, lr_config=None):
        super().__init__()

        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.rendering = rendering

        self.validation_dataset_mapping = validation_dataset_mapping
        self.validation_outputs = {}

        self.lr_config = {'start': 1e-4, 'end': 1e-5, 'iterations': 1e6} if lr_config is None else lr_config

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.rendering.parameters(), lr=self.lr_config['start'])
        self.scheduler = ExponentialLR(self.optimizer, gamma=(self.lr_config['end'] / self.lr_config['start']) ** (
                1 / self.lr_config['iterations']))
        return [self.optimizer], [self.scheduler]

    def on_train_batch_end(self, *args, **kwargs):
        # update learning rate and log
        if self.scheduler.get_last_lr()[0] > 5e-5:
            self.scheduler.step()
        self.log('Learning Rate', self.scheduler.get_last_lr()[0])

    def validation_epoch_end(self, outputs_list):
        if len(outputs_list) == 0:
            return  # skip invalid validation steps
        self.validation_outputs = {}  # reset validation outputs
        if isinstance(outputs_list[0], dict):
            outputs_list = [outputs_list]  # make list if only one validation dataset is used
        if len(outputs_list) == 0 or any([len(o) == 0 for o in outputs_list]):
            return  # skip invalid validation steps

        for i, outputs in enumerate(outputs_list):
            out_keys = outputs[0].keys()
            outputs = {k: torch.cat([o[k] for o in outputs]) for k in out_keys}
            self.validation_outputs[self.validation_dataset_mapping[i]] = outputs

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict, strict=False)
        self.validation_outputs = {}  # reset validation outputs


def save_state(sunerf: BaseSuNeRFModule, data_module: BaseDataModule, save_path):
    output_path = '/'.join(save_path.split('/')[0:-1])
    os.makedirs(output_path, exist_ok=True)
    torch.save({
        # sunerf  rendering module
        'rendering': sunerf.rendering,
        # data infor
        'data_config': data_module.config,
        # data scaling
        'Rs_per_ds': data_module.Rs_per_ds,
        'seconds_per_dt': data_module.seconds_per_dt,
        'ref_time': data_module.ref_time
    }, save_path)


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
        rays, time, target_image = batch['tracing']['rays'], batch['tracing']['time'], batch['tracing']['target_image']
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

        regularization_loss = outputs['regularization'].mean()  # suppress unconstrained regions
        loss = (self.lambda_image * (coarse_loss + fine_loss) +
                self.lambda_regularization * regularization_loss)
        #
        with torch.no_grad():
            psnr = -10. * torch.log10(fine_loss)

        # log results to WANDB
        self.log("loss", loss)
        self.log("train",
                 {'coarse': coarse_loss, 'fine': fine_loss,
                  'regularization': regularization_loss, 'psnr': psnr})

        return loss

    def validation_step(self, batch, batch_nb, **kwargs):
        dataloader_idx = kwargs['dataloader_idx'] if 'dataloader_idx' in kwargs else 0
        if dataloader_idx == 0:
            rays, time, target_image = batch['rays'], batch['time'], batch['target_image']
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
