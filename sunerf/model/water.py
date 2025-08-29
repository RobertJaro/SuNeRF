import numpy as np
import torch
from torch import nn

from sunerf.model.model import SirenNet
from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.util import jacobian
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.water import WaterRadiativeTransfer
from sunerf.train.scaling import ImageAsinhScaling, ImageLinearScaling, ImageLogScaling


class NEarthFModule(BaseSuNeRFModule):
    def __init__(self, meters_per_ds, seconds_per_dt,
                 instruments, sampling_config=None,
                 model_config=None, **kwargs):
        rendering_modules = {}
        scaling_modules = {}
        for instrument_config in instruments:
            instrument_config = instrument_config.copy()
            instrument_key = instrument_config.pop('key')
            instrument_type = instrument_config.pop('type')
            # rendering module
            rendering_config = instrument_config.pop('rendering', {})
            if instrument_type == 'default':
                rendering_modules[instrument_key] = WaterRadiativeTransfer()
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
        model = WaterVaporModel(**model_config)
        rendering = BasicRenderingModule(model=model,
                                         rendering_modules=rendering_modules,
                                         sampling_config={'type': 'flat_earth'})

        super().__init__(Rs_per_ds=meters_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        self.rendering_modules = rendering_modules
        self.model = model

        self.scaling_modules = nn.ModuleDict(scaling_modules)
        self.mse_loss = nn.MSELoss()


    def training_step(self, batch, batch_nb):
        dataset_batch = {k: v for k, v in batch.items() if k != 'random'}
        rendering_out = self.rendering(dataset_batch)

        model_out = rendering_out['model_out']

        query_points = rendering_out['query_points'].reshape(-1, 4)  # flatten query points for jacobian computation
        query_points = query_points[:4096]
        query_points.require_grad = True  # enable gradient computation for query points
        out  = self.rendering.model(query_points)
        rho = 10 ** out['log10_rho']
        jac_matrix = jacobian(rho, query_points)
        dRho_dz = jac_matrix[..., 2:3]  # assuming z is the third dimension
        gradient_loss = torch.relu(dRho_dz).pow(2).mean()

        instrument_image_diff = []

        for k in dataset_batch.keys():
            instrument_key = batch[k]['instrument']
            image_scaling = self.scaling_modules[instrument_key]

            model_image = model_out[k]['image']
            target_image = dataset_batch[k]['image']

            # scale images
            scaled_model_image = image_scaling(model_image)
            scaled_target_image = image_scaling(target_image)

            # backpropagation
            # optimize model
            image_diff = (scaled_model_image - scaled_target_image).pow(2).sum(-1)
            instrument_image_diff.append(image_diff)

        image_loss = torch.cat(instrument_image_diff).mean()

        assert torch.isnan(image_loss).sum() == 0, 'Invalid loss detected: image_loss'

        loss = image_loss + gradient_loss * 1e-3

        with torch.no_grad():
            psnr = -10. * torch.log10(image_loss)

        log_values = {'image': image_loss, 'psnr': psnr, 'gradient': gradient_loss}


        assert torch.isnan(loss).sum() == 0, 'Invalid loss detected: loss'
        # log results to WANDB
        self.log("loss", loss)
        self.log("train", log_values)

        return loss

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

            image_scaling = self.scaling_modules[instrument_key]
            target_image = image_scaling(image)
            model_image = image_scaling(model_image)

            return {'target_image': target_image,
                    'model_image': model_image}
        else:
            query_points = batch['query_points']
            model_out = self.model(query_points)
            return {'model_log10_rho': model_out['log10_rho'], 'true_log10_rho': batch['true_log10_rho'], 'query_points': query_points}


class WaterVaporModel(SirenNet):

    def __init__(self):
        super().__init__(4, 1, dim=256, w0_initial=30)

    def forward(self, coords):
        x = super().forward(coords)
        x = torch.sigmoid(x) * 4.7 - 2 # scale density to 10 ** -2 to 10 ** 3
        return {'log10_rho': x}