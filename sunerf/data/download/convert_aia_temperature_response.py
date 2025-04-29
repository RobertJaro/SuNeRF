import matplotlib.pyplot as plt
import numpy as np
from sunpy.io.special import read_genx

save_file = '/Users/rjarolim/PycharmProjects/SuNeRF/data/response_functions/aia_responses.npz'
genx_file = '/Users/rjarolim/PycharmProjects/SuNeRF/data/response_functions/aia_temp_resp.genx'

data = read_genx(genx_file)

wavelengths = [k for k in data.keys() if k != 'HEADER']
# response units = DN cm^5 s^-1 pix^-1
response = np.array([data[k]['TRESP'] for k in wavelengths])
temperature = data[wavelengths[0]]['LOGTE']

temperature = 10 ** temperature

plt.plot(temperature, response.T, label=wavelengths)
plt.legend()
plt.ylim(1e-29, 1e-23)
plt.loglog()
plt.show()

np.savez(save_file,
         temperature=temperature,
         response=response,
         wavelengths=wavelengths)
