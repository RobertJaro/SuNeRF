import copy

import numpy as np
from matplotlib import pyplot as plt
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

file = '/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_040_100/dcmer_100W_bang_0000_tB_stepnum_041.fits'

img = Map(file).data

cmap = cm.soholasco2.copy()

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
ax.imshow(img, norm='log', cmap=cmap, origin='lower')
ax.set_axis_off()

plt.tight_layout(pad=0)

fig.savefig('/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/cme_100.png', dpi=300, bbox_inches='tight',
            transparent=True)
plt.close(fig)