import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import torch.distributed as dist

class BaseSuNeRFModule(LightningModule):

    def __init__(self, Rs_per_ds, seconds_per_dt, rendering: nn.Module,
                 validation_dataset_mapping, lr_config=None):
        super().__init__()

        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.rendering = rendering

        self.validation_dataset_mapping = validation_dataset_mapping
        self.validation_outputs = {}
        self.validation_batches = {}

        self.lr_config = {'start': 1e-3, 'end': 1e-4, 'iterations': 1e6} if lr_config is None else lr_config

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

        self.rendering.on_train_batch_end(*args, **kwargs)

    def on_validation_epoch_start(self):
        self.validation_outputs = {}
        self.validation_batches = {}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        # Only rank-0 needs to keep outputs if you're running val on rank-0 only.
        if outputs is not None:
            # ensure CPU to keep GPU mem low
            cpu_out = {}
            dataset_key = self.validation_dataset_mapping.get(dataloader_idx, dataloader_idx)
            render_mode = batch.get('render_mode', None)
            if torch.is_tensor(render_mode):
                render_mode = render_mode.detach().cpu().view(-1)[0].item()
            for k, v in outputs.items():
                if not torch.is_tensor(v):
                    raise TypeError(
                        f"Validation output '{k}' for dataset '{dataset_key}' "
                        f"(dataloader_idx={dataloader_idx}, render_mode={render_mode}) "
                        f"must be a torch.Tensor, got {type(v).__name__}: {v!r}"
                    )
                cpu_out[k] = v.detach().cpu()
            cpu_out['dataset_idx'] = batch['dataset_idx'].detach().cpu()
            if dataloader_idx not in self.validation_batches:
                self.validation_batches[dataloader_idx] = []
            self.validation_batches[dataloader_idx].append(cpu_out)

    def on_validation_epoch_end(self):
        outputs_list = self.validation_batches
        if not outputs_list or len(outputs_list) == 0:
            return

        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            rank = dist.get_rank()
            world = dist.get_world_size()
            if rank == 0:
                obj_gather_list = [None] * world
                dist.gather_object(self.validation_batches, obj_gather_list, dst=0)
                # Merge dicts from all ranks
                merged_outputs = {}
                for rank_dict in obj_gather_list:
                    for dataloader_idx, batch_list in rank_dict.items():
                        if dataloader_idx not in merged_outputs:
                            merged_outputs[dataloader_idx] = []
                        merged_outputs[dataloader_idx].extend(batch_list)
                outputs_list = merged_outputs
            else:
                dist.gather_object(self.validation_batches, None, dst=0)
                return

        for dataloader_idx, outputs in outputs_list.items():
            # ---- reorder the list itself ----
            # get a single scalar lin_idx for each batch element
            # (use mean or first value if it's a vector)
            idxs = []
            for i, out in enumerate(outputs):
                lin_idx = out.pop('dataset_idx') # for sorting; discard for later steps
                if any([lin_idx == li for li, _ in idxs]):
                    continue # duplicated by DDP
                if lin_idx.ndim > 0:
                    lin_idx = lin_idx.view(-1)[0]  # take first sample in that batch
                idxs.append((int(lin_idx), i))
            # sort by the scalar lin_idx
            outputs = [outputs[i] for _, i in sorted(idxs, key=lambda x: x[0])]
            # ---- concatenate outputs ----
            out_keys = outputs[0].keys()
            outputs = {k: torch.cat([o[k] for o in outputs]) for k in out_keys}
            self.validation_outputs[self.validation_dataset_mapping[dataloader_idx]] = outputs

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict, strict=False)
        self.validation_outputs = {}  # reset validation outputs
