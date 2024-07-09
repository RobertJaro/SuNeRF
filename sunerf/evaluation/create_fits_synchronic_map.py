import datetime

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.map import Map, make_fitswcs_header


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


im = plt.imread('/Users/robert/PycharmProjects/sunerf/results/paper_plots/synchronic/sunerf_map_crop.jpg')
im = rgb2gray(im)

shape_out = im.shape

time = datetime.datetime(2012, 8, 30)
coord = SkyCoord(0, 0, unit=u.deg, frame="heliographic_carrington", observer='earth', obstime=time)
scale = [360 / shape_out[1], 180 / shape_out[0]] * u.deg / u.pix
header = make_fitswcs_header(shape_out, coord, scale=scale, projection_code="CAR")

Map(im, header).resample(u.Quantity((360 * 2, 180 * 2), u.pix)).save(
    '/Users/robert/PycharmProjects/sunerf/results/paper_plots/synchronic/sunerf_map.fits')
