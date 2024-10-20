import os

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from sunerf.data.loader.base_loader import BaseDataModule
from sunerf.model.model import AbsorptionModel
from sunerf.rendering.base_tracing import SuNeRFRendering
from sunerf.rendering.emission import EmissionRadiativeTransfer
from sunerf.rendering.plasma import PlasmaRadiativeTransfer
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
    state = {
        # sunerf  rendering module
        'rendering': sunerf.rendering,
        # data infor
        'data_config': data_module.config,
        # data scaling
        'Rs_per_ds': data_module.Rs_per_ds,
        'seconds_per_dt': data_module.seconds_per_dt,
        'ref_time': data_module.ref_time
    }
    if isinstance(sunerf, PlasmaSuNeRFModule):
        state['temperature_response'] = sunerf.temperature_response
        state['instrument_scaling'] = sunerf.instrument_scaling
        state['absorption_model'] = sunerf.absorption_model
    torch.save(state, save_path)


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
        loss = (self.lambda_image * (coarse_loss + fine_loss)
                # self.lambda_regularization * regularization_loss
                )
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


class PlasmaSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt,
                 image_scaling_config, temperature_response_config, opacity_config=None,
                 lambda_image=1.0, lambda_regularization=1.0,
                 sampling_config=None, hierarchical_sampling_config=None,
                 model_config=None, **kwargs):

        temperature_response = {k: np.load(c['file'])['response'] for k, c in temperature_response_config.items()}
        temperature_response = {k: nn.Parameter(torch.tensor(v.T, dtype=torch.float32), requires_grad=False) for k, v in
                                temperature_response.items()}
        # assert same number of temperature bins for all instruments
        assert len(set([v.shape[0] for v in
                        temperature_response.values()])) == 1, "Number of temperature bins must be the same for all instruments."

        temperature = np.load(list(temperature_response_config.values())[0]['file'])['temperature']
        log_T = torch.from_numpy(temperature).float()
        # setup rendering
        rendering = PlasmaRadiativeTransfer(log_T=log_T,
                                            Rs_per_ds=Rs_per_ds,
                                            sampling_config=sampling_config,
                                            hierarchical_sampling_config=hierarchical_sampling_config,
                                            model_config=model_config,
                                            absorption=opacity_config is not None)
        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        self.temperature_response = nn.ParameterDict(temperature_response)

        self.lambda_image = lambda_image
        self.lambda_regularization = lambda_regularization

        instrument_scaling = {k: nn.Parameter(torch.tensor(c['scaling'], dtype=torch.float32),
                                              requires_grad=c['learnable'])
                              for k, c in temperature_response_config.items()}
        self.instrument_scaling = nn.ParameterDict(instrument_scaling)

        image_scaling = {k: ImageAsinhScaling(**c) for k, c in image_scaling_config.items()}
        self.image_scaling = nn.ModuleDict(image_scaling)
        self.mse_loss = nn.MSELoss()

        self.absorption_model = AbsorptionModel(dim=16, n_layers=2)

    def configure_optimizers(self):
        params = (list(self.rendering.parameters()) +
                  list(self.instrument_scaling.parameters()) +
                  list(self.absorption_model.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.lr_config['start'])
        self.scheduler = ExponentialLR(self.optimizer, gamma=(self.lr_config['end'] / self.lr_config['start']) ** (
                1 / self.lr_config['iterations']))
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        instruments = batch.keys()

        total_loss = 0
        total_coarse_loss = 0
        total_fine_loss = 0
        for instrument_key in instruments:
            rays, time, target_image = (batch[instrument_key]['rays'],
                                        batch[instrument_key]['time'],
                                        batch[instrument_key]['image'])
            temperature_response = self.temperature_response[instrument_key]
            instrument_scaling = self.instrument_scaling[instrument_key]
            absorption_model = self.absorption_model

            rays_o, rays_d = rays[:, 0], rays[:, 1]

            outputs = self.rendering(rays_o, rays_d, time,
                                     temperature_response=temperature_response,
                                     instrument_scaling=instrument_scaling,
                                     absorption_model=absorption_model)

            # Check for any numerical issues.
            for k, v in outputs.items():
                assert not torch.isnan(v).any(), f"! [Numerical Alert] {k} contains NaN."
                assert not torch.isinf(v).any(), f"! [Numerical Alert] {k} contains Inf."

            out_coarse_image = outputs['coarse_image']
            out_fine_image = outputs['fine_image']

            image_scaling = self.image_scaling[instrument_key]
            # backpropagation
            target_image = image_scaling(target_image)
            # optimize coarse model
            coarse_image = image_scaling(out_coarse_image)
            # coarse_loss = (coarse_image - target_image) / target_image.mean(0, keepdim=True)
            # coarse_loss = coarse_loss.pow(2).mean()
            coarse_loss = self.mse_loss(coarse_image, target_image)
            # optimize fine model
            fine_image = image_scaling(out_fine_image)
            # fine_loss = (fine_image - target_image) / target_image.mean(0, keepdim=True)
            # fine_loss = fine_loss.pow(2).mean()
            fine_loss = self.mse_loss(fine_image, target_image)

            loss = self.lambda_image * (coarse_loss + fine_loss)
            total_loss += loss
            total_coarse_loss += coarse_loss
            total_fine_loss += fine_loss
        #
        with torch.no_grad():
            psnr = -10. * torch.log10(total_fine_loss)

        # log results to WANDB
        self.log("loss", total_loss)
        self.log("train",
                 {'coarse': total_coarse_loss, 'fine': total_fine_loss, 'psnr': psnr})

        return total_loss

    def validation_step(self, batch, batch_nb, *args):
        dataloader_idx = args[0] if len(args) > 0 else 0
        valid_ds_id = self.validation_dataset_mapping[dataloader_idx]
        if valid_ds_id == 'absorption':
            log_T, log_ne = batch['log_T'], batch['log_ne']
            absorption_input = torch.cat([log_ne, log_T], dim=-1)
            abs_out = self.absorption_model(absorption_input)
            return {**abs_out, 'log_T': log_T, 'log_ne': log_ne}
        else:
            rays, time, image = batch['rays'], batch['time'], batch['image']
            rays_o, rays_d = rays[:, 0], rays[:, 1]

            temperature_response = self.temperature_response[self.validation_dataset_mapping[dataloader_idx]]
            instrument_scaling = self.instrument_scaling[self.validation_dataset_mapping[dataloader_idx]]
            absorption_model = self.absorption_model

            outputs = self.rendering(rays_o, rays_d, time,
                                     temperature_response=temperature_response,
                                     instrument_scaling=instrument_scaling,
                                     absorption_model=absorption_model)

            image_scaling = self.image_scaling[self.validation_dataset_mapping[dataloader_idx]]

            image = torch.nan_to_num(image, nan=0.0)

            # print('IMG MIN MAX', image.min(), image.max())
            # print('IMG SCALING MIN MAX', image_scaling(image).min(), image_scaling(image).max())
            # print('FINE MIN MAX', outputs['fine_image'].min(), outputs['fine_image'].max())
            # print('FINE SCALING MIN MAX', image_scaling(outputs['fine_image']).min(), image_scaling(outputs['fine_image']).max())

            distance = rays_o.pow(2).sum(-1).pow(0.5)
            return {'target_image': image_scaling(image),
                    'fine_image': image_scaling(outputs['fine_image']),
                    'coarse_image': image_scaling(outputs['coarse_image']),
                    'mean_T': outputs['mean_T'], 'total_ne': outputs['total_ne'],
                    'height_map': outputs['height_map'],
                    'mean_absorption': outputs['mean_absorption'],
                    'z_vals_stratified': outputs['z_vals_stratified'],
                    'z_vals_hierarchical': outputs['z_vals_hierarchical'],
                    'distance': distance}

    def validation_epoch_end(self, *args, **kwargs):
        self.log(f'instrument_scaling',
                 {k: float(v.detach().cpu().numpy()) for k, v in self.instrument_scaling.items()})
        super().validation_epoch_end(*args, **kwargs)
