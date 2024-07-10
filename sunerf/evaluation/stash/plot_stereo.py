from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib import pyplot as plt
from sunpy.map import Map

from sunerf.data.utils import sdo_cmaps

stereo_a_map = Map('/mnt/nerf-data/stereo_2022_02_prep/304/2022-02-18T00:00:00_A.fits')
stereo_a_map = stereo_a_map.rotate(recenter=True)

sr = stereo_a_map.rsun_obs
stereo_a_map = stereo_a_map.submap(bottom_left=SkyCoord(-sr * 1.1, -sr * 1.1, frame=stereo_a_map.coordinate_frame),
                                   top_right=SkyCoord(sr * 1.1, sr * 1.1, frame=stereo_a_map.coordinate_frame))

norm = ImageNormalize(stretch=AsinhStretch(0.005), clip=True)
plt.imsave('/mnt/results/comparison_solo/original_stereo_a.jpg',
           norm(stereo_a_map.data), cmap=sdo_cmaps[304], origin='lower')
