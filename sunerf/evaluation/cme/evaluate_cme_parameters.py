import argparse
import glob
import os
from datetime import timedelta

import numpy as np
import scipy
from astropy import units as u
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from nf2.data.util import cartesian_to_spherical
from sklearn.linear_model import LinearRegression

from sunerf.data.date_util import normalize_datetime
from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.train.coordinate_transformation import spherical_to_cartesian

parser = argparse.ArgumentParser(description='Evaluate CME parameters')
parser.add_argument('--data_path', type=str, required=True, help='Path to density cube data files')
parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
parser.add_argument('--out_path', type=str, help='Path to output directory')
parser.add_argument('--max_radius', type=float, default=100.0, help='Maximum radius in solar radii')
parser.add_argument('--min_radius', type=float, default=30.0, help='Minimum radius in solar radii')
parser.add_argument('--min_longitude', type=float, default=None, help='Minimum longitude in degrees')
parser.add_argument('--max_longitude', type=float, default=None, help='Maximum longitude in degrees')
parser.add_argument('--min_latitude', type=float, default=-60.0, help='Minimum latitude in degrees')
parser.add_argument('--max_latitude', type=float, default=60.0, help='Maximum latitude in degrees')

args = parser.parse_args()
if args.out_path is None:
    args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'evaluation')
os.makedirs(args.out_path, exist_ok=True)

##########################################################
sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
seconds_per_dt = sunerf_loader.seconds_per_dt
ref_date = sunerf_loader.ref_date
print(ref_date)

##########################################################
max_radius = args.max_radius
min_radius = args.min_radius
min_longitude = args.min_longitude
max_longitude = args.max_longitude
min_latitude = args.min_latitude
max_latitude = args.max_latitude
Rs_per_ds = sunerf_loader.Rs_per_ds


##########################################################
# utility functions
def compute_center_of_mass(density, coords):
    spherical_coords = cartesian_to_spherical(coords)
    r, theta, phi = spherical_coords[..., 0], spherical_coords[..., 1], spherical_coords[..., 2]
    dr = np.gradient(r, axis=0)
    area_element = r ** 2 * np.sin(theta)
    total_mass = np.sum(density * area_element * dr)
    center_of_mass = np.sum(density[..., None] * area_element[..., None] * coords * dr[..., None],
                            axis=(0, 1, 2)) / total_mass
    center_of_mass = cartesian_to_spherical(center_of_mass)
    return center_of_mass, total_mass


times = []
center_of_mass_true = []
center_of_mass_pred = []
total_mass_true = []
total_mass_pred = []
slope_fit = []
center_of_mass_diff = []
correlation_coeffs = []
maes = []

date0 = parse("2010-04-03T09:04:00.000")
files = sorted(glob.glob(args.data_path))
files = files[:40]  # files[13:40]

for i, file in enumerate(files):
    ##########################################################
    o = scipy.io.readsav(file)
    time = date0 + timedelta(hours=float(o['this_time']))
    original_time = time
    time = normalize_datetime(time, seconds_per_dt, ref_date)

    density = o['dens'].astype(np.float32).T
    ph = o['ph1d'].astype(np.float32)
    r = o['r1d'].astype(np.float32)
    th = o['th1d'].astype(np.float32) - np.pi / 2

    # clip radius to 100 Rsun
    mask_r = (r < max_radius) & (r > min_radius)
    mark_th = np.ones_like(th, dtype=bool)
    mask_ph = np.ones_like(ph, dtype=bool)
    if min_longitude is not None:
        if min_longitude < 0:
            shifted = ((ph + np.pi) + np.deg2rad(min_longitude)) % (2 * np.pi)
            mask_ph = mask_ph & (shifted < 0)
        else:
            mask_ph = mask_ph & ((ph + np.pi) > np.deg2rad(min_longitude))
    if max_longitude is not None:
        mask_ph = mask_ph & ((ph + np.pi) < np.deg2rad(max_longitude))
    if max_latitude is not None:
        mark_th = mark_th & (th > np.deg2rad(min_latitude))
    if min_latitude is not None:
        mark_th = mark_th & (th < np.deg2rad(max_latitude))

    r = r[mask_r]
    th = th[mark_th]
    ph = ph[mask_ph]
    density = density[mask_r, :, :]
    density = density[:, mark_th, :]
    density = density[:, :, mask_ph]
    rho_true = density

    radius, theta, phi, t = np.meshgrid(r, th, ph, np.array([time]), indexing="ij")
    spherical_coords = np.stack([radius, theta, phi], axis=-1)

    cartesian_coords = spherical_to_cartesian(spherical_coords)
    cartesian_coords = cartesian_coords / Rs_per_ds
    x, y, z = cartesian_coords[..., 0], cartesian_coords[..., 1], cartesian_coords[..., 2]

    query_points = np.stack([x, y, z, t], axis=-1, dtype=np.float32)

    ##########################################################
    outputs = sunerf_loader.load_coords(query_points)

    rho_pred = outputs['rho']

    ##############################################################
    # plot slices

    rho_norm = LogNorm(vmin=1e1, vmax=1e3)

    lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - 0))
    r = spherical_coords[:, lat_idx, :, 0, 0]
    ph = spherical_coords[:, lat_idx, :, 0, 2]

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 5))

    ax = axs[0]
    z = rho_true[:, lat_idx, :]
    pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')
    fig.colorbar(pc, ax=ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]')
    ax.set_title("Ground-truth", va='bottom')

    ax = axs[1]
    z = rho_pred[:, lat_idx, :, 0, 0]
    pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')
    fig.colorbar(pc, ax=ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]')
    ax.set_title("SuNeRF", va='bottom')

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_path, f"lat_{i:03d}.jpg"), dpi=300)
    plt.close('all')

    ##############################################################
    # plot slices

    lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(135)))

    r = spherical_coords[:, :, lon_idx, 0, 0]
    th = spherical_coords[:, :, lon_idx, 0, 1]
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 5))

    ax = axs[0]
    z = rho_true[:, :, lon_idx]
    pc = ax.pcolormesh(th, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')  # , vmin=1e1, vmax=1e3)
    fig.colorbar(pc, ax=ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]')
    ax.set_title("SuNeRF", va='bottom')

    ax = axs[1]
    z = rho_pred[:, :, lon_idx, 0, 0]
    pc = ax.pcolormesh(th, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')  # , vmin=1e1, vmax=1e3)
    fig.colorbar(pc, ax=ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]')
    ax.set_title("Ground-truth", va='bottom')

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_path, f"lon_{i:03d}.jpg"), dpi=300)
    plt.close('all')

    ######################################################
    # check linear correlation
    flat_rho_true = rho_true.flatten()
    flat_rho_pred = rho_pred.flatten()

    model = LinearRegression(fit_intercept=False)
    model.fit(flat_rho_pred.reshape(-1, 1), flat_rho_true)
    print('Validation slope:', model.coef_[0])
    slope_fit.append(model.coef_[0])

    ##########################################################
    # compute center of mass

    com_true, tm_true = compute_center_of_mass(rho_true, cartesian_coords[:, :, :, 0, :] * Rs_per_ds)
    com_spherical_true = cartesian_to_spherical(com_true)
    com_spherical_true[..., 1] = np.rad2deg(com_spherical_true[..., 1])
    com_spherical_true[..., 2] = np.rad2deg(com_spherical_true[..., 2])
    com_pred, tm_pred = compute_center_of_mass(rho_pred[:, :, :, 0, 0], cartesian_coords[:, :, :, 0, :] * Rs_per_ds)
    com_spherical_pred = cartesian_to_spherical(com_pred)
    com_spherical_pred[..., 1] = np.rad2deg(com_spherical_pred[..., 1])
    com_spherical_pred[..., 2] = np.rad2deg(com_spherical_pred[..., 2])

    print('Predicted center of mass:', com_spherical_pred)
    print('Ground-truth center of mass:', com_spherical_true)

    ########################################################
    # statical values

    corr_coeff = np.corrcoef(flat_rho_true, flat_rho_pred)[0, 1]
    mae = np.abs(rho_true - rho_pred[:, :, :, 0, 0]).mean()

    # add to lists
    center_of_mass_true.append(com_spherical_true)
    center_of_mass_pred.append(com_spherical_pred)
    times.append(original_time)
    center_of_mass_diff.append(np.linalg.norm(com_true - com_pred, axis=-1))

    total_mass_true.append(tm_true)
    total_mass_pred.append(tm_pred)

    correlation_coeffs.append(corr_coeff)
    maes.append(mae)

print('Average slope:', np.mean(slope_fit))

times = np.array(times)
center_of_mass_true = np.array(center_of_mass_true)
center_of_mass_pred = np.array(center_of_mass_pred)
total_mass_true = np.array(total_mass_true)
total_mass_pred = np.array(total_mass_pred)
center_of_mass_diff = np.array(center_of_mass_diff)

ref_mass_true = total_mass_true[:5].mean()
ref_mass_pred = total_mass_pred[:5].mean()

cme_mass_true = total_mass_true - ref_mass_true
cme_mass_pred = total_mass_pred - ref_mass_pred

fig, axs = plt.subplots(2, 4, figsize=(10, 5))

ax = axs[0, 0]
ax.plot(times, center_of_mass_true[:, 0], '-o', label='Ground-truth')
ax.plot(times, center_of_mass_pred[:, 0], '-o', label='SuNeRF')
ax.set_ylabel('Radius [R$_\odot$]')
ax.set_xlabel('Time [UTC]')
ax.legend()
ax.set_title('Radius', va='bottom')

ax = axs[0, 1]
ax.plot(times, center_of_mass_true[:, 1], '-o', label='Ground-truth')
ax.plot(times, center_of_mass_pred[:, 1], '-o', label='SuNeRF')
ax.set_ylabel('Latitude [deg]')
ax.set_xlabel('Time [UTC]')
ax.legend()
ax.set_title('Theta', va='bottom')

ax = axs[0, 2]
ax.plot(times, center_of_mass_true[:, 2], '-o', label='Ground-truth')
ax.plot(times, center_of_mass_pred[:, 2], '-o', label='SuNeRF')
ax.set_ylabel('Longitude [deg]')
ax.set_xlabel('Time [UTC]')
ax.legend()
ax.set_title('Phi', va='bottom')

ax = axs[0, 3]
ax.plot(times, cme_mass_true, '-o', label='Ground-truth')
ax.plot(times, cme_mass_pred, '-o', label='SuNeRF')
ax.set_ylabel('Total mass [g]')
ax.set_xlabel('Time [UTC]')
ax.legend()
ax.set_title('Tota CME mass', va='bottom')

ax = axs[1, 0]
ax.plot(times, center_of_mass_diff, '-o', color='red')
ax.set_ylabel(r'$\Delta \vec{R}_\text{CoM}$ difference [R$_\odot$]')
ax.set_xlabel('Time [UTC]')
ax.set_title('Center of mass difference', va='bottom')

ax = axs[1, 1]
ax.plot(times, correlation_coeffs, '-o', color='red')
ax.set_ylabel('Correlation coefficient')
ax.set_xlabel('Time [UTC]')
ax.set_title('Correlation coefficient', va='bottom')

ax = axs[1, 2]
ax.plot(times, maes, '-o', color='red')
ax.set_ylabel('MAE [N$_\text{e}$ cm$^{-3}$]')
ax.set_xlabel('Time [UTC]')
ax.set_title('MAE', va='bottom')

axs[1, 3].axis('off')

# format x-axis
fig.autofmt_xdate()

fig.tight_layout()
fig.savefig(os.path.join(args.out_path, "center_of_mass.jpg"), dpi=300)
plt.close('all')

################################################
# compute average CME parameters
init_frame = 12
last_frame = 35


def compute_velocity(times, radius_data, init_frame, last_frame):
    """Compute CME velocity from time series data using linear regression."""
    time_seconds = np.array([(t - times[0]).total_seconds() for t in times])
    time_data = time_seconds[init_frame:last_frame]
    radius_data = radius_data[init_frame:last_frame]
    #
    velocity_model = LinearRegression()
    velocity_model.fit(time_data.reshape(-1, 1), radius_data)
    velocity = velocity_model.coef_[0] * u.Rsun / u.s
    velocity_km_s = velocity.to_value(u.km / u.s)
    #
    return velocity_km_s, time_data, velocity_model


# compute velocities for both true and predicted data
velocity_true, time_data, model_true = compute_velocity(times, center_of_mass_true[:, 0], init_frame, last_frame)
velocity_pred, _, model_pred = compute_velocity(times, center_of_mass_pred[:, 0], init_frame, last_frame)

# Plot velocity fits
plt.figure(figsize=(8, 6))
plt.plot(time_data / 3600, center_of_mass_true[init_frame:last_frame, 0], 'o', label='True CoM Radius')
plt.plot(time_data / 3600, center_of_mass_pred[init_frame:last_frame, 0], 'o', label='Predicted CoM Radius')
plt.plot(time_data / 3600, model_true.predict(time_data.reshape(-1, 1)),
         '-', label=f'True fit (v={velocity_true:.1f} km/s)')
plt.plot(time_data / 3600, model_pred.predict(time_data.reshape(-1, 1)),
         '-', label=f'Predicted fit (v={velocity_pred:.1f} km/s)')
plt.xlabel('Time since start [h]')
plt.ylabel('Radius [Rs]')
plt.title('CME Velocity from Center of Mass')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.out_path, "cme_velocity.jpg"), dpi=300)
plt.close('all')

print(f'True CME velocity: {velocity_true:.1f} km/s')
print(f'Predicted CME velocity: {velocity_pred:.1f} km/s')
