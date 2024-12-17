import torch
from sunerf.model.sunerf import PlasmaSuNeRFModule


class PlasmaSuNeRFModuleMHD(PlasmaSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt,
                 image_scaling_config, temperature_response_config,
                 lambda_image=1.0, lambda_regularization=1.0e-4, lambda_absorption=1.0e-4,
                 sampling_config=None, hierarchical_sampling_config=None,
                 model_config=None, shuffle_config=None, **kwargs):
        super().__init__(Rs_per_ds, seconds_per_dt,
                    image_scaling_config, temperature_response_config,
                    lambda_image=lambda_image, lambda_regularization=lambda_regularization, lambda_absorption=lambda_absorption,
                    sampling_config=sampling_config, hierarchical_sampling_config=hierarchical_sampling_config,
                    model_config=model_config, shuffle_config=shuffle_config, **kwargs)

    def training_step(self, batch, batch_nb):
        query_points_time, density, temperature = batch['psi']
        query_points_time.requires_grad = True

        coarse_raw = self.rendering.coarse_model(query_points_time)
        fine_raw = self.rendering.fine_model(query_points_time)

        coarse_loss_density = (coarse_raw['total_log_ne'].squeeze() - torch.log10(density)).pow(2).mean(-1)
        coarse_loss_temperature = (coarse_raw['mean_log_T'].squeeze() - torch.log10(temperature)).pow(2).mean(-1)

        fine_loss_density = (fine_raw['total_log_ne'].squeeze() - torch.log10(density)).pow(2).mean(-1)
        fine_loss_temperature = (fine_raw['mean_log_T'].squeeze() - torch.log10(temperature)).pow(2).mean(-1)
        
        loss = coarse_loss_density + fine_loss_density + coarse_loss_temperature + fine_loss_temperature

        # log results to WANDB
        self.log("loss", loss)
        self.log("train",
                 {'coarse_density': coarse_loss_density, 'fine_density': fine_loss_density,
                  'coarse_temperature': coarse_loss_temperature, 'fine_temperature': fine_loss_temperature})

        return loss        
    
    def validation_step(self, batch, batch_nb, *args):
        pass

    def validation_epoch_end(self, *args, **kwargs):
        pass
