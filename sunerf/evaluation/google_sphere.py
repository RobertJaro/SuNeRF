import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from astropy.nddata import block_reduce
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

result_path = '/mnt/results/google_sphere_jpg'
os.makedirs(result_path, exist_ok=True)
sunerf_map = plt.imread('/mnt/results/topo_map/sunerf_map_crop.jpg')
sunerf_map = block_reduce(sunerf_map, (5, 5, 1), func=np.mean)

coords_lat = np.linspace(-90, 90, sunerf_map.shape[0])
coords_lon = np.linspace(0, 360, sunerf_map.shape[1])

coords = np.stack(np.meshgrid(coords_lat, coords_lon), -1)

phi, theta = (coords[..., 0] + 90) * np.pi / 180, coords[..., 1] * np.pi / 180
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta) - 1
z = np.cos(phi) - 1

fig = plt.figure(figsize=(10, 10))
spec = fig.add_gridspec(nrows=1, ncols=1, left=0.02,
                        right=0.98, bottom=0.02, top=0.98, wspace=0.00)
ax = fig.add_subplot(spec[:, :], projection=Axes3D.name)
ax.set_axis_off()
ax.plot_surface(x, y, z, facecolors=sunerf_map.astype(np.float32).transpose((1, 0, 2)) / 255,
                antialiased=False, shade=False, rstride=1, cstride=1)
limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz']);
ax.set_box_aspect(np.ptp(limits, axis=1))

margin = 50

for i in tqdm(range(0, 360)):
    ax.view_init(elev=0, azim=-i, roll=0, vertical_axis='z')
    filename = '/home/benoit_tremblay_23/000_%03d.png' % i
    plt.savefig(filename, transparent=True)

    img = np.array(Image.open(filename))

    if i == 0:
        ny_sun, nx_sun, c_sun = img.shape
        where_sun = img[:, :, 3]
        where_sun_x = np.sum(where_sun, axis=1)
        where_sun_y = np.sum(where_sun, axis=0)

        if np.amax(where_sun_x) != np.amax(where_sun_y):
            print('Aspect ratio is wrong.')
            exit()
        else:
            radius_sun = int(np.amax(where_sun_x / 255) / 2)

        x_sun = np.where(where_sun_x > 0)[0][0]
        y_sun = np.where(where_sun_y > 0)[0][0]
        x_center = int(nx_sun / 2) - radius_sun
        y_center = int(ny_sun / 2) - radius_sun
    else:
        img = np.array(Image.open(filename))

    img = np.roll(img, (y_sun - y_center, x_sun - x_center, 0), axis=(0, 1, 2))
    out = Image.fromarray(
        img[y_center - margin:y_center + 2 * radius_sun + margin, x_center - margin:x_center + 2 * radius_sun + margin,
        :])
    out.save(filename)

for i in tqdm(range(0, 360)):
    ax.view_init(elev=i, azim=0, roll=0, vertical_axis='z')
    filename = '/home/benoit_tremblay_23/%03d_000.png' % i
    plt.savefig(filename, transparent=True)
    img = np.array(Image.open(filename))

    img = np.roll(img, (y_sun - y_center, x_sun - x_center, 0), axis=(0, 1, 2))
    out = Image.fromarray(
        img[y_center - margin:y_center + 2 * radius_sun + margin, x_center - margin:x_center + 2 * radius_sun + margin,
        :])
    out.save(filename)

plt.close(fig)

shutil.make_archive(result_path, 'zip', result_path)
