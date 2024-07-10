import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch
from pyhdf.SD import SD, SDC
from sunpy.visualization.colormaps import cm

filename = '/Users/robert/PycharmProjects/sunerf/data/maps_r101_euv_3205.hdf'

ds = SD(filename, SDC.READ)
ds.datasets()

img = ds.select('Data-Set-2').get()
img[img < 0] = np.NAN
img = np.roll(img, 1600, axis=1)

x_interp = np.linspace(-1, 1, 1600)
x = np.sin(np.linspace(-np.pi / 2, np.pi / 2, 1600))
img_interp = np.stack([np.interp(x, x_interp, img[:, i]) for i in range(img.shape[1])], 1)

fig = plt.figure(figsize=(16, 8))
plt.imshow(10 ** img_interp, extent=(-180, 180, -90, 90), cmap=cm.sdoaia193, origin='lower',
           norm=ImageNormalize(stretch=AsinhStretch(0.005)))
plt.xlabel('Carrington Longitude', fontsize='x-large')
plt.ylabel('Carrington Latitude', fontsize='x-large')
fig.savefig('/Users/robert/PycharmProjects/sunerf/results/paper_plots/synchronic/psi.png', dpi=300, transparent=True)
plt.close(fig)
