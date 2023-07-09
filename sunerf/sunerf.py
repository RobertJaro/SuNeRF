import argparse
import os

import logging
import numpy as np
import torch
import yaml
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger
from skimage.metrics import structural_similarity
from torch.optim.lr_scheduler import ExponentialLR

from sunerf.data.utils import sdo_cmaps
from sunerf.evaluation.callback import plot_samples, log_overview
from sunerf.train.model import init_models
from sunerf.train.sampling import sample_to_solar_surface, sample_non_uniform_box
from sunerf.train.volume_render import nerf_forward
from sunerf.utilities.data_loader import NeRFDataModule, unnormalize_datetime


class SuNeRFModule(LightningModule):
    def __init__(self, hparams, cmap):
        super().__init__()
        self.cmap = cmap

        for key in hparams.keys():
            self.hparams[key] = hparams[key]
        self.lambda_regularization = self.hparams['Lambda']['regularization']

        self.start_iter = 0  # TODO: Update this based on loading the checkpoint
        self.n_samples_hierarchical = self.hparams['Hierarchical sampling']['n_samples_hierarchical']

        # We bundle the kwargs for various functions to pass all at once.
        self.kwargs_sample_stratified = {
            'n_samples': self.hparams['Stratified sampling']['n_samples'],
            'perturb': self.hparams['Stratified sampling']['perturb'],
        }

        # Pick and setup sampler
        if self.hparams['Stratified sampling']['non_uniform_sampling']:
            self.sample_stratified = sample_non_uniform_box
            self.kwargs_sample_stratified['grid_exponent'] = self.hparams['Stratified sampling']['grid_exponent']
        else:
            self.sample_stratified = sample_to_solar_surface

        self.kwargs_sample_hierarchical = {
            'perturb': self.hparams['Stratified sampling']['perturb']
        }

        # Model Loading
        self.coarse_model, self.fine_model, self.encode, self.model_params = init_models(
            d_input=self.hparams['Encoders']['d_input'],
            d_output=self.hparams['Model']['d_output'],
            n_freqs=self.hparams['Encoders']['n_freqs'],
            n_layers=self.hparams['Model']['n_layers'],
            d_filter=self.hparams['Model']['d_filter'],
            log_space=self.hparams['Encoders']['log_space'],
            use_fine_model=self.hparams['Model']['use_fine_model'],
            skip=self.hparams['Model']['skip'],
        )

        near, far = self.hparams['Stratified sampling']['near'], self.hparams['Stratified sampling']['far']
        self.sampling_kwargs = {'sample_stratified': self.sample_stratified,
                                'kwargs_sample_stratified': self.kwargs_sample_stratified,
                                'n_samples_hierarchical': self.n_samples_hierarchical,
                                'near': near, 'far': far,
                                'kwargs_sample_hierarchical': self.kwargs_sample_hierarchical}
        self.encoder_kwargs = {'d_input': self.hparams['Encoders']['d_input'],
                               'n_freqs': self.hparams['Encoders']['n_freqs'],
                               'log_space': self.hparams['Encoders']['log_space'], }

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model_params, lr=float(self.hparams["Optimizer"]["lr"]))
        self.scheduler = ExponentialLR(self.optimizer, gamma=(5e-5 / 5e-4) ** (1 / 1e6), )  # decay over 1e6 iterations

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_nb):
        rays, time, target_img = batch
        rays_o, rays_d = rays[:, 0], rays[:, 1]
        # Run one iteration of TinyNeRF and get the rendered filtergrams.
        outputs = nerf_forward(rays_o, rays_d, time, self.coarse_model, self.fine_model,
                               encoding_fn=self.encode, **self.sampling_kwargs)

        # Check for any numerical issues.
        for k, v in outputs.items():
            assert not torch.isnan(v).any(), f"! [Numerical Alert] {k} contains NaN."
            assert not torch.isinf(v).any(), f"! [Numerical Alert] {k} contains Inf."

        # backpropagation
        pred_img = outputs['channel_map']
        fine_loss = torch.nn.functional.mse_loss(pred_img, target_img) # optimize fine model
        coarse_loss = torch.nn.functional.mse_loss(outputs['channel_map_0'], target_img)  # optimize coarse model
        regularization_loss = outputs['regularization'].mean() # suppress unconstrained regions
        loss = fine_loss + coarse_loss + self.lambda_regularization * regularization_loss
        #
        with torch.no_grad():
            psnr = -10. * torch.log10(fine_loss)

        # log results to WANDB
        self.log("train/loss", loss)
        self.log("Training Loss", {'coarse': coarse_loss, 'fine': fine_loss, 'regularization': regularization_loss, 'total': loss})
        self.log("Training PSNR", psnr)

        # update learning rate and log
        if self.scheduler.get_last_lr()[0] > 5e-5:
            self.scheduler.step()
        self.log('Learning Rate', self.scheduler.get_last_lr()[0])

        return {'loss': loss, 'train_psnrs': psnr}

    def validation_step(self, batch, batch_nb):
        rays, time, target_img = batch
        rays_o, rays_d = rays[:, 0], rays[:, 1]

        outputs = nerf_forward(rays_o, rays_d, time, self.coarse_model, self.fine_model,
                               encoding_fn=self.encode, **self.sampling_kwargs)

        distance = rays_o.pow(2).sum(-1).pow(0.5).mean()
        return {'target_img': target_img,
                'channel_map': outputs['channel_map'],
                'channel_map_coarse': outputs['channel_map_0'],
                'height_map': outputs['height_map'],
                'absorption_map': outputs['absorption_map'],
                'z_vals_stratified': outputs['z_vals_stratified'],
                'z_vals_hierarchical': outputs['z_vals_hierarchical'],
                'distance': distance}

    def validation_epoch_end(self, outputs):
        target_img = torch.cat([o['target_img'] for o in outputs])
        channel_map = torch.cat([o['channel_map'] for o in outputs])
        channel_map_coarse = torch.cat([o['channel_map_coarse'] for o in outputs])
        height_map = torch.cat([o['height_map'] for o in outputs])
        absorption_map = torch.cat([o['absorption_map'] for o in outputs])
        z_vals_stratified = torch.cat([o['z_vals_stratified'] for o in outputs])
        z_vals_hierarchical = torch.cat([o['z_vals_hierarchical'] for o in outputs])

        wh = int(np.sqrt(target_img.shape[0]))
        target_img = target_img.view(wh, wh, target_img.shape[1]).cpu().numpy()
        channel_map = channel_map.view(wh, wh, channel_map.shape[1]).cpu().numpy()
        channel_map_coarse = channel_map_coarse.view(wh, wh, channel_map_coarse.shape[1]).cpu().numpy()
        height_map = height_map.view(wh, wh).cpu().numpy()
        absorption_map = absorption_map.view(wh, wh).cpu().numpy()
        z_vals_stratified = z_vals_stratified.view(wh, wh, -1).cpu().numpy()
        z_vals_hierarchical = z_vals_hierarchical.view(wh, wh, -1).cpu().numpy()
        distance = outputs[0]['distance'].cpu().numpy().mean()

        # TODO move plotting to separate plot callback
        plot_samples(channel_map, channel_map_coarse, height_map, absorption_map, target_img, z_vals_stratified,
                     z_vals_hierarchical, distance=distance, cmap=self.cmap)

        val_loss = ((channel_map - target_img) ** 2).mean()
        val_ssim = structural_similarity(target_img[..., 0], channel_map[..., 0], data_range=1)
        val_psnr = -10. * np.log10(val_loss)
        self.log("Validation Loss", val_loss)
        self.log("Validation PSNR", val_psnr)
        self.log("Validation SSIM", val_ssim)

        return {'progress_bar': {'val_loss': val_loss,
                                 'val_psnr': val_psnr,
                                 'val_ssim': val_ssim},
                'log': {'val/loss': val_loss, 'val/psnr': val_psnr, 'val/ssim': val_ssim}}


def save_state(sunerf: SuNeRFModule, data_module: NeRFDataModule, save_path):
    output_path = '/'.join(save_path.split('/')[0:-1])
    os.makedirs(output_path, exist_ok=True)
    torch.save({'fine_model': sunerf.fine_model, 'coarse_model': sunerf.coarse_model,
                'wavelength': data_module.wavelength,
                'sampling_kwargs': sunerf.sampling_kwargs, 'encoder_kwargs': sunerf.encoder_kwargs,
                'test_kwargs': data_module.test_kwargs, 'config': config_data,
                'start_time': unnormalize_datetime(min(data_module.times)),
                'end_time': unnormalize_datetime(max(data_module.times))},
               save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
    N_GPUS = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--path_to_save', type=str, required=True)
    parser.add_argument('--n_epochs', default=100, type=int, help='number of training epochs.')
    parser.add_argument('--hyperparameters', default='../../config/hyperparams.yaml', type=str)
    parser.add_argument('--train', default='../../config/train.yaml', type=str)
    parser.add_argument('--resume_from_checkpoint', default=None, type=str, required=False)
    parser.add_argument('--wandb_project', default='sunerf-v3', type=str, required=False)
    parser.add_argument('--wandb_name', default=None, type=str, required=False)
    parser.add_argument('--wandb_id', default=None, type=str, required=False)
    args = parser.parse_args()

    config_data = {'data_path': args.data_path}
    with open(args.hyperparameters, 'r') as stream:
        config_data.update(yaml.load(stream, Loader=yaml.SafeLoader))
    with open(args.train, 'r') as stream:
        config_data.update(yaml.load(stream, Loader=yaml.SafeLoader))

    data_module = NeRFDataModule(config_data)
    cmap = sdo_cmaps[data_module.wavelength]  # set global colormap

    sunerf = SuNeRFModule(config_data, cmap)

    # manually laod --> automatic load checkpoint causes error
    chk_path = args.resume_from_checkpoint
    if chk_path is not None:
        logging.info(f'Load checkpoint: {chk_path}')
        sunerf.load_state_dict(torch.load(chk_path, map_location=torch.device('cpu'))['state_dict'])

    checkpoint_callback = ModelCheckpoint(dirpath=args.path_to_save,
                                          save_top_k=5,
                                          monitor='train/loss',
                                          every_n_train_steps=config_data["Training"]["log_every_n_steps"])

    logger = WandbLogger(project=args.wandb_project, offline=False, entity="4pi-euv",
                         name=args.wandb_name, id=args.wandb_id)

    save_path = os.path.join(args.path_to_save, 'save_state.snf')
    os.makedirs(args.path_to_save, exist_ok=True)
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path))

    logging.info('Initialize trainer')
    trainer = Trainer(max_epochs=args.n_epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy='dp' if N_GPUS > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=-1,  # validate all points to check the first image
                      val_check_interval=config_data["Training"]["log_every_n_steps"],
                      gradient_clip_val=0.5,
                      callbacks=[checkpoint_callback, save_callback])

    log_overview(data_module.images, data_module.poses, data_module.times, cmap)

    logging.info('Start model training')
    trainer.fit(sunerf, data_module)
    trainer.save_checkpoint(os.path.join(args.path_to_save, 'final.ckpt'))
