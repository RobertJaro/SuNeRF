import math

import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
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
        self._validation_output_keys = {}
        self._validation_output_prefixes = {}
        self._validation_output_filter_enabled = False

        self.lr_config = {'start': 1e-3, 'end': 1e-4, 'iterations': 1e6} if lr_config is None else lr_config

    def configure_optimizers(self):
        start = float(self.lr_config['start'])
        end = float(self.lr_config['end'])
        iterations_float = float(self.lr_config['iterations'])
        iterations = int(iterations_float)
        if not math.isfinite(start) or not math.isfinite(end) or start <= 0 or end <= 0:
            raise ValueError("Learning-rate schedule 'start' and 'end' must be finite and positive.")
        if not math.isfinite(iterations_float) or iterations_float != iterations or iterations <= 0:
            raise ValueError("Learning-rate schedule 'iterations' must be a positive integer.")

        self.optimizer = torch.optim.Adam(self.parameters(), lr=start)
        end_factor = end / start
        log_end_factor = math.log(end_factor)

        def lr_factor(step):
            # Lightning advances this scheduler once per optimizer step.  Returning
            # the exact endpoint after ``iterations`` prevents continued decay (or
            # growth) when training runs longer than the configured schedule.
            if step >= iterations:
                return end_factor
            return math.exp(log_end_factor * max(step, 0) / iterations)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_factor)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'interval': 'step',
                'frequency': 1,
                'name': 'Learning Rate',
            },
        }

    def on_train_batch_end(self, *args, **kwargs):
        # Lightning owns scheduler stepping through the step-interval configuration
        # returned by ``configure_optimizers``.  Stepping here as well advances it
        # twice on some Lightning versions and bypasses its checkpoint state.
        self.log('Learning Rate', self.optimizer.param_groups[0]['lr'])

        self.rendering.on_train_batch_end(*args, **kwargs)

    def on_validation_epoch_start(self):
        self.validation_outputs = {}
        self.validation_batches = {}

    def register_validation_output_requirements(self, dataset_key, keys=(), prefixes=()):
        """Register the output fields consumed by callbacks for one validation dataset.

        Multiple callbacks can share a dataset, so requirements are accumulated.  The
        registration is done by :class:`BaseCallback` during Lightning setup, before
        validation starts.
        """
        self._validation_output_filter_enabled = True
        self._validation_output_keys.setdefault(dataset_key, set()).update(keys)
        self._validation_output_prefixes.setdefault(dataset_key, set()).update(prefixes)

    def _filter_validation_outputs(self, dataset_key, outputs):
        if not self._validation_output_filter_enabled:
            return outputs

        keys = self._validation_output_keys.get(dataset_key, set())
        prefixes = self._validation_output_prefixes.get(dataset_key, set())
        return {
            key: value
            for key, value in outputs.items()
            if key in keys or any(key.startswith(prefix) for prefix in prefixes)
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        if outputs is None:
            return

        dataset_key = self.validation_dataset_mapping.get(dataloader_idx, dataloader_idx)
        outputs = self._filter_validation_outputs(dataset_key, outputs)
        if not outputs:
            return

        # Detach only callback-consumed fields and stage them on the CPU.  In DDP,
        # non-zero ranks keep only their local shard until the typed tensor gather at
        # epoch end; Python-object serialization is deliberately avoided.
        cpu_out = {}
        render_mode = batch.get('render_mode', None)
        if torch.is_tensor(render_mode):
            render_mode = render_mode.detach().cpu().view(-1)[0].item()
        for key, value in outputs.items():
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Validation output '{key}' for dataset '{dataset_key}' "
                    f"(dataloader_idx={dataloader_idx}, render_mode={render_mode}) "
                    f"must be a torch.Tensor, got {type(value).__name__}: {value!r}"
                )
            cpu_out[key] = value.detach().cpu()
        cpu_out['dataset_idx'] = batch['dataset_idx'].detach().cpu()
        self.validation_batches.setdefault(dataloader_idx, []).append(cpu_out)

    @staticmethod
    def _merge_validation_batches(outputs):
        """Restore dataset order, drop DDP padding duplicates, and concatenate."""
        if not outputs:
            return None

        unique = {}
        for output in outputs:
            dataset_idx = int(output['dataset_idx'].view(-1)[0].item())
            unique.setdefault(dataset_idx, output)

        ordered = [unique[index] for index in sorted(unique)]
        output_keys = tuple(key for key in ordered[0] if key != 'dataset_idx')
        for output in ordered:
            keys = tuple(key for key in output if key != 'dataset_idx')
            if keys != output_keys:
                raise RuntimeError(
                    "Validation output fields changed between batches: "
                    f"expected {output_keys}, got {keys}."
                )
        return {
            key: torch.cat([output[key] for output in ordered], dim=0)
            for key in output_keys
        }

    def _collective_device(self):
        backend = str(dist.get_backend()).lower()
        return self.device if 'nccl' in backend else torch.device('cpu')

    @staticmethod
    def _all_gather_sizes(local_size, device):
        size = torch.tensor([local_size], dtype=torch.long, device=device)
        gathered = [torch.empty_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, size)
        return [int(value.item()) for value in gathered]

    @staticmethod
    def _all_gather_padded(tensor, sizes, device):
        """Gather a variable-length first dimension with typed collectives."""
        tensor = tensor.to(device=device, non_blocking=True).contiguous()
        max_size = max(sizes)
        padded_shape = (max_size,) + tuple(tensor.shape[1:])
        padded = torch.zeros(padded_shape, dtype=tensor.dtype, device=device)
        if tensor.shape[0]:
            padded[:tensor.shape[0]].copy_(tensor)
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
        if dist.get_rank() != 0:
            return None
        return [value[:size].cpu() for value, size in zip(gathered, sizes)]

    def _gather_validation_batches(self, outputs):
        """Gather one dataloader's CPU shards as tensors and rebuild its batches."""
        field_names = tuple(key for key in outputs[0] if key != 'dataset_idx')
        for output in outputs:
            fields = tuple(key for key in output if key != 'dataset_idx')
            if fields != field_names:
                raise RuntimeError(
                    "Validation output fields changed between batches: "
                    f"expected {field_names}, got {fields}."
                )

        batch_indices = torch.tensor(
            [int(output['dataset_idx'].view(-1)[0].item()) for output in outputs],
            dtype=torch.long,
        )
        batch_lengths = torch.tensor(
            [int(output[field_names[0]].shape[0]) for output in outputs],
            dtype=torch.long,
        )
        for output, expected_length in zip(outputs, batch_lengths.tolist()):
            for key in field_names:
                if output[key].ndim == 0 or output[key].shape[0] != expected_length:
                    raise RuntimeError(
                        f"Validation field '{key}' does not share the batch dimension "
                        f"({tuple(output[key].shape)} versus {expected_length})."
                    )

        device = self._collective_device()
        batch_counts = self._all_gather_sizes(len(outputs), device)
        sample_counts = self._all_gather_sizes(int(batch_lengths.sum().item()), device)
        indices_by_rank = self._all_gather_padded(batch_indices, batch_counts, device)
        lengths_by_rank = self._all_gather_padded(batch_lengths, batch_counts, device)

        records_by_rank = None
        if dist.get_rank() == 0:
            records_by_rank = [
                [
                    {'dataset_idx': index.view(1)}
                    for index in rank_indices
                ]
                for rank_indices in indices_by_rank
            ]

        for key in field_names:
            local_values = torch.cat([output[key] for output in outputs], dim=0)
            values_by_rank = self._all_gather_padded(local_values, sample_counts, device)
            if dist.get_rank() != 0:
                continue
            for rank_records, rank_values, rank_lengths in zip(
                    records_by_rank, values_by_rank, lengths_by_rank):
                offset = 0
                for record, length_tensor in zip(rank_records, rank_lengths):
                    length = int(length_tensor.item())
                    record[key] = rank_values[offset:offset + length]
                    offset += length

        if dist.get_rank() != 0:
            return None
        return [record for rank_records in records_by_rank for record in rank_records]

    def on_validation_epoch_end(self):
        if not self.validation_batches:
            return

        is_distributed = dist.is_available() and dist.is_initialized()
        for dataloader_idx, outputs in self.validation_batches.items():
            if is_distributed:
                outputs = self._gather_validation_batches(outputs)
                if dist.get_rank() != 0:
                    continue
            merged = self._merge_validation_batches(outputs)
            if merged is not None:
                self.validation_outputs[self.validation_dataset_mapping[dataloader_idx]] = merged

        # Non-zero ranks never run rank-zero callbacks and should not retain a local
        # validation shard after the collective completes.
        self.validation_batches = {}

    def on_validation_end(self):
        """Release callback tensors as soon as all validation callbacks finish."""
        self.validation_outputs = {}
        self.validation_batches = {}

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict, strict=False)
        self.validation_outputs = {}  # reset validation outputs
