import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

save_file = '/Users/rjarolim/PycharmProjects/SuNeRF/data/response_functions/stereo_behind_responses.npz'
geny_file = '/Users/rjarolim/PycharmProjects/SuNeRF/data/response_functions/behind_sre_chianti2_fludra_mazzotta_002.geny'

sra_file = '/Users/rjarolim/PycharmProjects/SuNeRF/data/response_functions/ahead_sra_001.geny'
sra_data = readsav(sra_file)

data = readsav(geny_file)

temperature = data['p0'][0][5]  # (61, )

# 6 = electrons; 7 = photons
# response units = 1e44 EM
response = data['p0'][0][7]  # 4, 4, 61
response = response[:, 1, :]

wavelengths = data['p0'][0][8][:, 0]

print('Wavelengths', wavelengths)
print('Modes', data['p0'][0][9][0, :], f"select --> {data['p0'][0][9][0, 1]}")

plt.plot(temperature, response.T, label=wavelengths)
plt.legend()
plt.loglog()
plt.show()

np.savez(save_file,
         temperature=temperature,
         response=response,
         wavelengths=wavelengths)
