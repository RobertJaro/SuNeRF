from datetime import datetime

import drms
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib.colors import Normalize
from sunpy.map import Map, all_coordinates_from_map

from nf2.data.download import donwload_ds

euv_map = Map('/Users/rjarolim/PycharmProjects/SuNeRF/data/aia.lev1_euv_12s.2023-02-24T203410Z.171.image_lev1.fits')
coords = all_coordinates_from_map(euv_map)
radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / euv_map.rsun_obs

data = euv_map.data
alpha = 1.1 - (radius - 1) / 0.1
alpha[radius <= 1] = 1
alpha[alpha < 0] = 0
alpha[alpha > 1] = 1

cmap = plt.get_cmap('sdoaia171')
norm = ImageNormalize(data, vmin=1, vmax=2e4, stretch=AsinhStretch(a=0.01))
img = cmap(norm(data))
img[..., 3] = alpha
plt.imsave('/Users/rjarolim/PycharmProjects/SuNeRF/data/aia.lev1_euv_12s.2023-02-24T203410Z.171.image_lev1.png', img, origin='lower')


hmi_map = Map('/Users/rjarolim/PycharmProjects/SuNeRF/data/hmi.B_720s.20230224_203600_TAI.Br.fits')

coords = all_coordinates_from_map(hmi_map)
radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / hmi_map.rsun_obs

data = hmi_map.data
alpha = np.ones_like(data)
alpha[radius > 1] = 0

cmap = plt.get_cmap('gray')
norm = Normalize(vmin=-1000, vmax=1000)
img = cmap(norm(data))
img[..., 3] = alpha
plt.imsave('/Users/rjarolim/PycharmProjects/SuNeRF/data/hmi.B_720s.20230224_203600_TAI.Br.png', img, origin='lower')

kso_map = Map('/Users/rjarolim/PycharmProjects/SuNeRF/data/kanz_halph_fi_20230224_100505.fts')
coords = all_coordinates_from_map(kso_map)

radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / kso_map.rsun_obs
data = kso_map.data
alpha = np.ones_like(data)
alpha[radius > 1] = 0
cmap = plt.get_cmap('gray')
norm = Normalize(vmin=0)
img = cmap(norm(data))
img[..., 3] = alpha

plt.imsave('/Users/rjarolim/PycharmProjects/SuNeRF/data/kanz_halph_fi_20230224_100505.png', img, origin='lower')
