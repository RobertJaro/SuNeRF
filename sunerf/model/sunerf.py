import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR


class BaseSuNeRFModule(LightningModule):

    def __init__(self, Rs_per_ds, seconds_per_dt, rendering: nn.Module,
                 validation_dataset_mapping, lr_config=None):
        super().__init__()

        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.rendering = rendering

        self.validation_dataset_mapping = validation_dataset_mapping
        self.validation_outputs = {}

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

        if self.rendering.shuffler is not None:
            self.rendering.shuffler.on_train_batch_end()

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


