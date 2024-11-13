import os.path

import numpy as np
from astropy.io.fits import getdata, getheader
from matplotlib import pyplot as plt

out_path = '/glade/work/rjarolim/sunerf/response'
file_174 = '/glade/work/rjarolim/data/sunerf/temperature_response/gof_fsi_174_filter25_sun_coronal_2021_chianti.abund_chianti.ioneq_synthetic.fits'
file_304 = '/glade/work/rjarolim/data/sunerf/temperature_response/gof_fsi_304_filter26_sun_coronal_2021_chianti.abund_chianti.ioneq_synthetic.fits'

eui_174 = getdata(file_174)
eui_304 = getdata(file_304)

eui_174_header = getheader(file_174)
eui_304_header = getheader(file_304)

ne = 10 ** np.linspace(8, 18, eui_174.shape[0])
temperature = 10 ** np.linspace(4, 8, eui_174.shape[1])

eui_174_G = eui_174 / ne[:, None] ** 2 * 1e10
eui_304_G = eui_304 / ne[:, None] ** 2 * 1e10

target_ne = 1e9
ne_idx = np.argmin(np.abs(ne - target_ne))
eui_174_response = eui_174_G[ne_idx]
eui_304_response = eui_304_G[ne_idx]


temperature_response = {'fsi_174': eui_174_response, 'fsi_304': eui_304_response}

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for key, response in temperature_response.items():
    ax.plot(temperature, response, label=key)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Response function')
ax.set_title('AIA Response Functions')
ax.legend()
plt.savefig(os.path.join(out_path, 'eui_response_functions.png'))
plt.show()

np.savez(os.path.join(out_path, 'eui_response_functions.npz'), temperature=temperature, **temperature_response)