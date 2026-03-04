import argparse
import glob
import multiprocessing
import os

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
import numpy as np
import torch
from sunpy.visualization.colormaps import cm
from torch import nn
from tqdm import tqdm

from sunerf.data.loader.base_loader import MapDataLoader
from sunerf.model.model import SirenModel


class BackgroundModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.stray_light = SirenModel(in_dim=2, out_dim=1, dim=128, n_layers=4,
                                      encoding_config={'type': 'default', 'w0': 30.})
        self.starfield = SirenModel(in_dim=3, out_dim=1, dim=128, n_layers=4,
                                    encoding_config={'type': 'default', 'w0': 30.})
        self.softplus = nn.Softplus()

    def forward(self, coords, rays_d):
        x = coords[..., 0:2]

        starfield = self.softplus(self.starfield(rays_d))
        stray_light = torch.exp(self.stray_light(x))

        background = starfield + stray_light
        return {'background': background,
                'starfield': starfield,
                'stray_light': stray_light}


def _get_norm(data, log_scale=True):
    valid = np.isfinite(data)
    if not np.any(valid):
        return Normalize(vmin=0.0, vmax=1.0)
    d = data[valid]
    if log_scale:
        d = d[d > 0]
        if d.size == 0:
            return Normalize(vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))
        vmin = np.nanpercentile(d, 5)
        vmax = np.nanpercentile(d, 99.5)
        vmin = max(float(vmin), 1e-12)
        vmax = max(float(vmax), vmin * 1.01)
        return LogNorm(vmin=vmin, vmax=vmax)
    vmin = float(np.nanpercentile(d, 1))
    vmax = float(np.nanpercentile(d, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return Normalize(vmin=vmin, vmax=vmax)


def _plot_validation(epoch, out, gt_frame, output_path):
    nan_mask = np.isnan(gt_frame)

    gt_frame = gt_frame.copy()
    fit = out['background'].copy()
    starfield_component = out['starfield'].copy()
    stray_light_component = out['stray_light'].copy()

    gt_frame[nan_mask] = np.nan
    fit[nan_mask] = np.nan
    starfield_component[nan_mask] = np.nan
    stray_light_component[nan_mask] = np.nan

    common_cmap = cm.soholasco2
    residual = gt_frame - fit

    ref_norm = _get_norm(gt_frame, log_scale=True)
    finite_residual = residual[np.isfinite(residual)]
    if finite_residual.size > 0:
        vmax = float(np.nanpercentile(np.abs(finite_residual), 99))
        vmax = max(vmax, 1e-8)
    else:
        vmax = 1.0
    diff_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))

    ax = axs[0, 0]
    im = ax.imshow(gt_frame, origin='lower', cmap=common_cmap, norm=ref_norm)
    ax.set_title('GT frame')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[0, 1]
    im = ax.imshow(fit, origin='lower', cmap=common_cmap, norm=ref_norm)
    ax.set_title('Fit frame')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[0, 2]
    im = ax.imshow(residual, origin='lower', cmap='coolwarm', norm=diff_norm)
    ax.set_title('GT - fit')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1, 0]
    ax.set_axis_off()

    ax = axs[1, 1]
    im = ax.imshow(stray_light_component, origin='lower', cmap=common_cmap, norm=ref_norm)
    ax.set_title('Component: stray light')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1, 2]
    im = ax.imshow(starfield_component, origin='lower', cmap=common_cmap, norm=ref_norm)
    ax.set_title('Component: starfield')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Validation epoch {epoch}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_background_subtracted(epoch, gt_frame, background_component, output_path):
    nan_mask = np.isnan(gt_frame)
    gt = gt_frame.copy()
    bg_subtracted = gt - background_component

    gt[nan_mask] = np.nan
    bg_subtracted[nan_mask] = np.nan
    bg_subtracted[bg_subtracted <= 0] = np.nan

    ref_norm = _get_norm(gt, log_scale=True)
    common_cmap = cm.soholasco2

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axs[0]
    im = ax.imshow(gt, origin='lower', cmap=common_cmap, norm=ref_norm)
    ax.set_title('GT')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1]
    im = ax.imshow(bg_subtracted, origin='lower', cmap=common_cmap, norm=ref_norm)
    ax.set_title('GT - background')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Background-subtracted GT (epoch {epoch})')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _batched_model_inference(model_fn, coords_cpu, rays_d_cpu, batch_size, device):
    flat_coords = coords_cpu.reshape(-1, coords_cpu.shape[-1])
    flat_rays_d = rays_d_cpu.reshape(-1, rays_d_cpu.shape[-1])
    outputs = {}
    with torch.no_grad():
        for start in range(0, flat_coords.shape[0], batch_size):
            end = min(start + batch_size, flat_coords.shape[0])
            coords_batch = flat_coords[start:end].to(device)
            rays_d_batch = flat_rays_d[start:end].to(device)
            batch_out = model_fn(coords_batch, rays_d_batch)
            for key, value in batch_out.items():
                outputs.setdefault(key, []).append(value.detach().cpu())
    reshaped = {}
    for key, chunks in outputs.items():
        values = torch.cat(chunks, dim=0).squeeze(-1)
        reshaped[key] = values.reshape(*coords_cpu.shape[:-1]).numpy()
    return reshaped


def _build_arg_parser():
    parser = argparse.ArgumentParser(description='Fit and validate background correction model.')
    parser.add_argument('--file_path', type=str, help='Input FITS glob pattern.')
    parser.add_argument('--out_dir', type=str, help='Output directory for checkpoints and plots.')
    parser.add_argument('--pixel_norm', type=float, default=256.0)
    parser.add_argument('--image_norm', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=131072)
    parser.add_argument('--validation_batch_size', type=int, default=131072)
    parser.add_argument('--validation_every', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--linear_loss_weight', type=float, default=1.0)
    parser.add_argument('--bg_nonneg_log_weight', type=float, default=1.0e-3)
    parser.add_argument('--bg_negative_weight', type=float, default=1.0)
    parser.add_argument('--bg_min_residual_log10', type=float, default=-4.0)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint file path. Defaults to <out_dir>/background_model.pt')
    return parser


def main():
    args = _build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(args.file_path))
    if len(files) == 0:
        raise FileNotFoundError(f'No files found for pattern: {args.file_path}')

    loader = MapDataLoader(Rs_per_ds=1, reference_frame='inertial', azimuthal_equidistant=False)

    with multiprocessing.Pool(args.num_workers) as pool:
        data = [v for v in tqdm(pool.imap(loader.load, files), total=len(files), desc='Loading files')]

    image = np.stack([d['image'] for d in data], axis=0)
    rays_d = np.stack([d['rays'][..., 1, :] for d in data], axis=0)
    time = np.array([d['time'] for d in data])

    # basic pre-processing --> 5% subtraction
    minimum_mask = np.nanpercentile(image, 5, axis=(1, 2), keepdims=True)
    image = image - minimum_mask
    image = np.clip(image, a_min=1e-15, a_max=None)

    t, x, y = np.mgrid[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]
    image_coordinates = np.stack([x, y, t], axis=-1).astype(np.float32)
    image_coordinates[..., 0] -= image.shape[1] / 2
    image_coordinates[..., 1] -= image.shape[2] / 2
    image_coordinates[..., :2] /= args.pixel_norm
    image_coordinates[..., 2] /= image.shape[0]

    image = image / args.image_norm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nan_mask = ~np.isnan(image)
    image_tensor = torch.tensor(image[nan_mask], dtype=torch.float32)
    coords_tensor = torch.tensor(image_coordinates[nan_mask, :], dtype=torch.float32)
    rays_d_tensor = torch.tensor(rays_d[nan_mask, :], dtype=torch.float32)

    validation_frame_idx = image.shape[0] // 2
    validation_gt_frame = image[validation_frame_idx]
    validation_coords_tensor = torch.tensor(image_coordinates[validation_frame_idx], dtype=torch.float32)
    validation_rays_d_tensor = torch.tensor(rays_d[validation_frame_idx], dtype=torch.float32)

    model = BackgroundModel().to(device)
    model_fn = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = args.checkpoint_path or os.path.join(args.out_dir, 'background_model.pt')
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = int(checkpoint['epoch']) + 1
            print(f'Loaded checkpoint from {checkpoint_path} (resume at epoch {start_epoch})')
        elif isinstance(checkpoint, nn.Module):
            model.load_state_dict(checkpoint.state_dict())
            print(f'Loaded model weights from {checkpoint_path}')
        else:
            raise RuntimeError(f'Unsupported checkpoint format in {checkpoint_path}')

    n_training_samples = image_tensor.shape[0]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_data_loss_sum = 0.0
        epoch_linear_loss_sum = 0.0
        epoch_bg_nonneg_reg_sum = 0.0
        permutation = torch.randperm(n_training_samples)

        progress = tqdm(
            range(0, n_training_samples, args.train_batch_size),
            desc=f'Epoch {epoch + 1}/{args.epochs}',
            leave=False,
        )
        for start in progress:
            end = min(start + args.train_batch_size, n_training_samples)
            idx = permutation[start:end]

            optimizer.zero_grad(set_to_none=True)
            coords_batch = coords_tensor[idx].to(device)
            rays_d_batch = rays_d_tensor[idx].to(device)
            image_batch = image_tensor[idx].to(device)

            output = model_fn(coords_batch, rays_d_batch)
            background = output['background'].squeeze(-1)
            log_background = torch.log(background + 1e-8)
            log_image = torch.log(image_batch + 1e-8)
            data_loss = torch.mean((log_background - log_image) ** 2)
            linear_loss = torch.mean((background - image_batch) ** 2)
            bg_residual = image_batch - background
            log_bg_residual = torch.log10(torch.clamp(bg_residual, min=1e-12))
            bg_small_residual_reg = torch.mean(torch.relu(args.bg_min_residual_log10 - log_bg_residual) ** 2)
            bg_negative_reg = torch.mean(torch.relu(-bg_residual) ** 2)
            bg_nonneg_reg = args.bg_nonneg_log_weight * bg_small_residual_reg + args.bg_negative_weight * bg_negative_reg
            loss = data_loss + args.linear_loss_weight * linear_loss + bg_nonneg_reg

            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * (end - start)
            epoch_data_loss_sum += data_loss.item() * (end - start)
            epoch_linear_loss_sum += linear_loss.item() * (end - start)
            epoch_bg_nonneg_reg_sum += bg_nonneg_reg.item() * (end - start)
            progress.set_postfix(
                batch_loss=f'{loss.item():.3e}',
                log_loss=f'{data_loss.item():.3e}',
                lin_loss=f'{linear_loss.item():.3e}',
                bg_nonneg=f'{bg_nonneg_reg.item():.3e}',
            )

        epoch_loss = epoch_loss_sum / n_training_samples
        epoch_data_loss = epoch_data_loss_sum / n_training_samples
        epoch_linear_loss = epoch_linear_loss_sum / n_training_samples
        epoch_bg_nonneg_reg = epoch_bg_nonneg_reg_sum / n_training_samples

        print(
            f'Epoch {epoch}, Loss: {epoch_loss:.6e}, '
            f'LogLoss: {epoch_data_loss:.6e}, LinLoss: {epoch_linear_loss:.6e}, '
            f'BgNonNeg: {epoch_bg_nonneg_reg:.6e}'
        )

        if epoch % args.validation_every == 0:
            model.eval()
            validation_out = _batched_model_inference(
                model_fn=model_fn,
                coords_cpu=validation_coords_tensor,
                rays_d_cpu=validation_rays_d_tensor,
                batch_size=args.validation_batch_size,
                device=device,
            )
            _plot_validation(
                epoch=epoch,
                out=validation_out,
                gt_frame=validation_gt_frame,
                output_path=os.path.join(args.out_dir, f'validation_{epoch:04d}.jpg'),
            )
            _plot_background_subtracted(
                epoch=epoch,
                gt_frame=validation_gt_frame,
                background_component=validation_out['background'],
                output_path=os.path.join(args.out_dir, f'validation_bg_subtracted_{epoch:04d}.jpg'),
            )
            full_background = validation_out['background'] * args.image_norm + minimum_mask
            np.save(
                os.path.join(args.out_dir, f'background_{epoch:04d}.npy'),
                full_background,
            )
            model.train()
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                checkpoint_path,
            )


if __name__ == '__main__':
    main()
