import argparse
import glob
import os

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from sunpy.coordinates import frames
from sunpy.map import make_fitswcs_header
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from sunerf.data.loader.psi import PSICubeDataset
from sunerf.data.psi.read_psi import read_PSI
from sunerf.model.model import PlasmaModel
from sunerf.rendering.plasma import PlasmaRadiativeTransfer

parser = argparse.ArgumentParser()
parser.add_argument('--temperature_response_file', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--Rs_per_ds', type=float, default=1)
parser.add_argument('--seconds_per_dt', type=float, default=86400)
parser.add_argument('--work_directory', type=str)
parser.add_argument('--out_path', type=str)
parser.add_argument('--reload', action='store_true')
args = parser.parse_args()

out_path = args.out_path
os.makedirs(out_path, exist_ok=True)

model_path = os.path.join(out_path, 'save_state.snf')

# get temperature response function
temperature_response = np.load(args.temperature_response_file)
temperature = temperature_response['temperature']
log_T = torch.from_numpy(temperature).float()

# create dataset
dataset = PSICubeDataset(data_path=args.data_path, Rs_per_ds=args.Rs_per_ds, seconds_per_dt=args.seconds_per_dt,
                         work_directory=args.work_directory, batch_size=int(2 ** 18))

# create model or load from previous state
if os.path.exists(model_path) and not args.reload:
    model = torch.load(model_path)['rendering'].fine_model
    print('Loaded model from previous state')
else:
    model = PlasmaModel(log_T=log_T, decay_distance=2.5)

parallel_model = DataParallel(model)

# create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)
parallel_model.train()
parallel_model.to(device)

# prepare validation data
rho_file = sorted(glob.glob(os.path.join(args.data_path, 'rho', '*.h5')))[0]
t_file = sorted(glob.glob(os.path.join(args.data_path, 't', '*.h5')))[0]
psi_data = read_PSI(rho_file, t_file)

cartesian_coords = psi_data['cartesian_coordinates'][:, :, 0]
validation_rho_true = psi_data['rho'][:, :, 0]
validation_T_true = psi_data['T'][:, :, 0]
#
coords = np.zeros((*cartesian_coords.shape[:-1], 4), dtype=np.float32)
coords[..., :3] = cartesian_coords
coords[..., 3] = psi_data['time']
validation_tensor = torch.from_numpy(coords).to(device)

########################################################################################################################
# plot coordinates of validation data
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

ax = axs[0]
im = ax.imshow(cartesian_coords[..., 0], origin='lower', cmap='viridis')
ax.set_title('x')
fig.colorbar(im, ax=ax)

ax = axs[1]
im = ax.imshow(cartesian_coords[..., 1], origin='lower', cmap='viridis')
ax.set_title('y')
fig.colorbar(im, ax=ax)

ax = axs[2]
im = ax.imshow(cartesian_coords[..., 2], origin='lower', cmap='viridis')
ax.set_title('z')
fig.colorbar(im, ax=ax)

fig.tight_layout()
plt.savefig(os.path.join(out_path, 'coords.jpg'))
plt.close()

########################################################################################################################
# save rendering model function
temperature_response_config = [{
    'file': '/glade/work/rjarolim/sunerf/response/aia_interpolated.npz',
    'scaling': 0.0,
    'learnable': False,
    'instruments': ['PSI']}]
sampling_config = {'type': 'spherical', 'distance': 2.5}

# dummy module for rendering
rendering = PlasmaRadiativeTransfer(temperature_response_config=temperature_response_config,
                                    Rs_per_ds=args.Rs_per_ds,
                                    sampling_config=sampling_config, absorption_config={'type': 'constant'})
# create mock WCS
obs = SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU, frame=frames.HeliographicStonyhurst, obstime=dataset.ref_time)
reference_coord = SkyCoord(0 * u.deg, 0 * u.deg, obstime=dataset.ref_time, observer=obs,
                           frame=frames.Helioprojective)
mock_data = np.zeros((512, 512))
scale = [2400 / 256, 2400 / 256] * u.arcsec / u.pix
wcs = make_fitswcs_header(mock_data, reference_coord, scale=scale)

data_config = {'PSI': {'times': dataset.times, 'image_shape': (512, 512), 'wcs': wcs}}


def save_model():
    # overwrite previous coarse and fine model
    rendering.coarse_model.load_state_dict(model.state_dict())
    rendering.fine_model.load_state_dict(model.state_dict())
    state = {
        # sunerf  rendering module
        'rendering': rendering,
        # data infor
        'data_config': data_config,
        # data scaling
        'Rs_per_ds': args.Rs_per_ds,
        'seconds_per_dt': args.seconds_per_dt,
        'ref_time': dataset.ref_time,
    }
    torch.save(state, model_path)


########################################################################################################################
# start training
epochs = 100
for epoch in range(epochs):
    total_loss = []
    for batch in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        log_rho_true = batch['log_rho'].to(device)
        log_T_true = batch['log_T'].to(device)
        coords = batch['coords'].to(device)
        #
        model_out = parallel_model(coords)
        log_ne = model_out['total_log_ne']
        mean_log_T = model_out['mean_log_T']
        rho_loss = (log_rho_true - log_ne).pow(2).mean()
        T_loss = (log_T_true - mean_log_T).pow(2).mean()
        loss = rho_loss + T_loss
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
    print(f'Epoch {epoch}, loss: {torch.stack(total_loss).mean()}')

    ########################## Validation
    model_validation_out = parallel_model(validation_tensor)
    log_ne_pred = model_validation_out['total_log_ne'].detach().cpu().numpy()
    mean_log_T_pred = model_validation_out['mean_log_T'].detach().cpu().numpy()

    # plot validation image
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    ax = axs[0, 0]
    im = ax.imshow(validation_rho_true, origin='lower', cmap='viridis', norm='log')
    ax.set_title('rho_true')
    fig.colorbar(im, ax=ax)

    ax = axs[0, 1]
    im = ax.imshow(validation_T_true, origin='lower', cmap='inferno', norm='log')
    ax.set_title('T_true')
    fig.colorbar(im, ax=ax)

    ax = axs[1, 0]
    im = ax.imshow(10 ** log_ne_pred, origin='lower', cmap='viridis', norm='log')
    ax.set_title('rho_pred')
    fig.colorbar(im, ax=ax)

    ax = axs[1, 1]
    im = ax.imshow(10 ** mean_log_T_pred, origin='lower', cmap='inferno', norm='log')
    ax.set_title('T_pred')
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    plt.savefig(os.path.join(out_path, f'validation_{epoch:03d}.jpg'))
    plt.close()

    ########################## Save model
    save_model()
