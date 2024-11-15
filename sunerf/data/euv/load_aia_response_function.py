import os.path

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import readsav

out_path = '/glade/work/rjarolim/sunerf/response'
euv_resp = readsav('/glade/work/rjarolim/sunerf/response/aia_euv_resp.sav')

temperature = 10 ** euv_resp['aiaresp'][0][7]
response = euv_resp['aiaresp'][0][8]
channel_keys = euv_resp['aiaresp'][0][5].astype(str)

temperature_response = {k: response[i] for i, k in enumerate(channel_keys)}

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for key, response in temperature_response.items():
    ax.plot(temperature, response, label=key)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Response function')
ax.set_title('AIA Response Functions')
ax.legend()
plt.savefig(os.path.join(out_path, 'aia_response_functions.png'))
plt.show()

np.savez(os.path.join(out_path, 'aia_response_functions.npz'), temperature=temperature, **temperature_response)