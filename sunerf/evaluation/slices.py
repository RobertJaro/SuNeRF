import argparse
import os
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt, axes
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.coordinates.utils import GreatArc

from sunerf.evaluation.loader import SuNeRFLoader

# Argument parsing
parser = argparse.ArgumentParser('Create video of ecliptic and polar views')
parser.add_argument('--chk_path', type=str)
parser.add_argument('--video_path', type=str)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=int(2 ** 12))
args = parser.parse_args()

# Create output directory
os.makedirs(args.video_path, exist_ok=True)

# Initialize loader
loader = SuNeRFLoader(args.chk_path)

# Define ranges
center_lon = 132 + 90
target_lon = 132 + 90
ref_time = datetime(2023, 4, 11)

latitude_range = np.arange(-60, 60.5, 0.5) * u.deg
longitude_range = np.arange(center_lon - 60, center_lon + 60.5, 0.5) * u.deg
# longitude_range = np.linspace(0, 360, 361) * u.deg
radius = np.linspace(1, 1.4, 100) * u.solRad

# Load data
out = loader.load_spherical(longitude_range=longitude_range,
                            radius_range=radius,
                            latitude_range=latitude_range,
                            time=ref_time)
ne = out['ne']
log_T = loader.log_T_range
mean_log_T = out['mean_log_T']

# Load observer image
img_out = loader.load_observer_image(lat=0 * u.deg, lon=center_lon * u.deg, time=ref_time, instrument_key='AIA_FD',
                                     resolution=(256, 256) * u.pix)
total_ne = out['total_ne'][..., 0, 0]

# Plot integrated ne map
extent = [longitude_range[0].to_value(u.deg), longitude_range[-1].to_value(u.deg), latitude_range[0].to_value(u.deg),
          latitude_range[-1].to_value(u.deg)]

ne_map = np.sum(total_ne, axis=-1)
em_bin_1 = ne[..., (log_T > 5.0) & (log_T < 6.0)].sum((2, 3, 4))
em_bin_2 = ne[..., (log_T > 6.0) & (log_T < 7.0)].sum((2, 3, 4))

# plotting settings
aia_193_norm = ImageNormalize(vmin=0, vmax=0.5, stretch=AsinhStretch(0.01))

fig = plt.figure(figsize=(12, 8))

ax = plt.subplot(1, 3, 1)
im = ax.imshow(em_bin_1, cmap='jet', origin='upper', extent=extent, norm=LogNorm(vmin=5e2))
ax.set_title('EM 5.0 - 6.0 [cm$^{-2}$]')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05, label='')
plt.colorbar(im, cax=cax)
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.axvline(target_lon, color='black', linestyle='--')

ax = plt.subplot(1, 3, 2)
im = ax.imshow(em_bin_2, cmap='jet', origin='upper', extent=extent, norm=LogNorm(vmin=5e2))
ax.set_title('EM 6.0 - 7.0 [cm$^{-2}$]')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.axvline(target_lon, color='black', linestyle='--')

map_193 = img_out['maps'][193]
ax = plt.subplot(1, 3, 3, projection=map_193)
im = ax.imshow(map_193.data, origin='lower', cmap='sdoaia193', norm=aia_193_norm)

# draw red box around the region
bottom_left = SkyCoord(lon=longitude_range[0], lat=latitude_range[0], frame=frames.HeliographicCarrington,
                    observer=map_193.observer_coordinate)
top_left = SkyCoord(lon=longitude_range[-1], lat=latitude_range[0], frame=frames.HeliographicCarrington,
                        observer=map_193.observer_coordinate)
top_right = SkyCoord(lon=longitude_range[-1], lat=latitude_range[-1], frame=frames.HeliographicCarrington,
                        observer=map_193.observer_coordinate)
bottom_right = SkyCoord(lon=longitude_range[0], lat=latitude_range[-1], frame=frames.HeliographicCarrington,
                        observer=map_193.observer_coordinate)
great_arc = GreatArc(top_left, top_right)
ax.plot_coord(great_arc.coordinates(), color='red')
great_arc = GreatArc(top_right, bottom_right)
ax.plot_coord(great_arc.coordinates(), color='red')
great_arc = GreatArc(bottom_right, bottom_left)
ax.plot_coord(great_arc.coordinates(), color='red')
great_arc = GreatArc(bottom_left, top_left)
ax.plot_coord(great_arc.coordinates(), color='red')
##
# plot arc of longitude
start_coord = SkyCoord(lon=target_lon * u.deg, lat=latitude_range[0], frame=frames.HeliographicCarrington,
                       observer=map_193.observer_coordinate)
end_coord = SkyCoord(lon=target_lon * u.deg, lat=latitude_range[-1], frame=frames.HeliographicCarrington,
                     observer=map_193.observer_coordinate)
great_arc = GreatArc(start_coord, end_coord)
ax.plot_coord(great_arc.coordinates(), color='black', linestyle='--')
map_193.draw_grid(grid_spacing=20 * u.deg, color='black', alpha=0.5)
#
ax.set_title(r'AIA 193 Å [DN/s]')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=axes.Axes)
cbar = plt.colorbar(im, cax=cax)
cbar.set_ticks([0.5, 0.1, 0])

ax.set_xlabel('X [arcsec]')
ax.set_ylabel('Y [arcsec]')

plt.tight_layout()
plt.savefig(os.path.join(args.video_path, 'total_ne_map.png'), dpi=300, transparent=True)
plt.close(fig)


def _prep_polar(ax):
    ax.set_theta_zero_location('S')
    # ax.set_theta_direction(-1)
    # theta limits
    ax.set_thetamin(latitude_range[0].to_value(u.rad) + np.pi/2)
    ax.set_thetamax(latitude_range[-1].to_value(u.rad) + np.pi/2)
    # theta ticks
    x_ticks = np.linspace(latitude_range[0], latitude_range[-1], 7)
    ax.set_xticks(x_ticks.to_value(u.rad) + np.pi / 2)
    ax.set_xticklabels(x_ticks.to_value(u.deg))
    # radius ticks
    ax.set_rlim(2 - radius[-1].value, radius[-1].value)
    ax.set_yticks([1, 1.1, 1.2, 1.3, 1.4])


s_map = map_193

s_map.center.transform_to(frames.HeliographicCarrington)

T_norm = LogNorm(vmin=10 ** 5.0, vmax=10 ** 6.5)
ne_norm = LogNorm(vmin=1, vmax=500)

theta, r = np.meshgrid(np.pi / 2 - latitude_range.to_value(u.rad), radius.to_value(u.solRad))

lon_idx = np.argmin(np.abs(longitude_range - target_lon * u.deg))



lon_ne = ne[:, lon_idx, :, 0]
lon_total_ne = np.sum(lon_ne, axis=-1)
lon_mean_T = 10 ** mean_log_T[:, lon_idx, :, 0, 0]
#
plt.clf()

fig = plt.figure(figsize=(8, 4))
#
ax = fig.add_subplot(121, projection='polar')
_prep_polar(ax)
im = ax.pcolormesh(theta, r, lon_total_ne.T, norm=ne_norm, cmap='jet', edgecolors='face')
# ax.set_title('Total electron density')
plt.colorbar(im, ax=ax, orientation='vertical', label='$N_e$ [cm$^{-3}$]')
#
ax = fig.add_subplot(122, projection='polar')
_prep_polar(ax)
alpha = np.clip((lon_total_ne - 5) / 100, 0, 1)
im = ax.pcolormesh(theta, r, lon_mean_T.T, cmap='inferno', norm=T_norm, edgecolors='face', alpha=alpha.T)
# ax.set_title('Mean temperature')
plt.colorbar(im, ax=ax, orientation='vertical', label='T [K]')
#
# plt.suptitle(f'Longitude: {target_lon:.1f}')
plt.tight_layout()
plt.savefig(os.path.join(args.video_path, f'lon_slice_{target_lon:03.2f}.png'), dpi=300, transparent=True)
plt.close(fig)