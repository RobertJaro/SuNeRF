import torch
from torch import nn
import glob
import os
from sunerf.data.mhd.psi_io import rdhdf_3d
import cupy as cp
from scipy.interpolate import RegularGridInterpolator as rgi
from cupyx.scipy.interpolate import RegularGridInterpolator as rgi_gpu
import numpy as np
import h5py as h5
import gc

class MHDModel(nn.Module):
    r""" Interpolation of MHD model such that it behaves like a trained NeRF
    """
    def __init__(self, log_T, data_path, device=None):
        """_summary_

        Parameters
        ----------
        data_path : str
        """            
        super().__init__()

        self.log_T = log_T
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        self.data_path = data_path
        self.density_files = sorted(glob.glob(os.path.join(data_path, 'rho', '*.h5')))
        self.temperature_files = sorted(glob.glob(os.path.join(data_path, 't', '*.h5')))
        self.ffirst = int(self.density_files[0].split('00')[1].split('.h5')[0])  # rho002531.h5
        self.flast = int(self.density_files[-1].split('00')[1].split('.h5')[0])


    def interp(self, phi, theta, r, f, var, method='linear', fill_value=None):
        """Interpolation of MHD data

        Args:
            phi (float): azimuthal angle
            theta (float): polar angle
            r (float): radial distance
            f (int): frame number
            var (str): variable to interpolate
            method (str, optional): interpolation type. Defaults to 'linear'.
            fill_value (float, optional): fill value. Defaults to 1e-10.

        Returns:
            torch.Tensor: interpolated data
        """

        # Read data
        # log_memory_usage("+++++++++ Open file")
        # r_mhd, th_mhd, phi_mhd, data = rdhdf_3d(os.path.join(self.data_path, var, f'{var}00{1813}.h5'))

        h5file = h5.File(os.path.join(self.data_path, var, f'{var}00{f}.h5'), 'r')
        data = h5file['Data']
        dims = data.shape
        ndims = np.ndim(data)

        # Clip radius
        r[r<1.0] = 1.0

        # Get the scales if they exist:
        for i in range(0, ndims):
            if i == 0:
                if len(h5file['Data'].dims[0].keys()) != 0:
                    r_mhd = h5file['Data'].dims[0][0]
            elif i == 1:
                if len(h5file['Data'].dims[1].keys()) != 0:
                    th_mhd = h5file['Data'].dims[1][0]
            elif i == 2:
                if len(h5file['Data'].dims[2].keys()) != 0:
                    phi_mhd = h5file['Data'].dims[2][0]

        r_mhd = np.array(r_mhd)
        th_mhd = np.array(th_mhd)
        phi_mhd = np.array(phi_mhd)
        data = np.array(data)
        h5file.close()
        del h5file
        gc.collect()

        # r_mhd, th_mhd, phi_mhd, data = self.r_mhd, self.th_mhd, self.phi_mhd, self.data
        # print(f'{var}00{f}.h5', np.amin(data), np.amax(data), np.percentile(data, 1), np.percentile(data, 99), np.percentile(data, 5), np.percentile(data, 95))
        # Filter data
        # log_memory_usage("+++++++++ File opened")
        fill_value = np.median(data[np.where(data > 0)])
        data[np.where(data < 0)] = fill_value

        # Interpolation based on device
        if self.device.type == 'cuda':
            f1_interp = rgi_gpu((cp.array(phi_mhd), cp.array(th_mhd), cp.array(r_mhd)), cp.array(data),
                                method=method, bounds_error=False, fill_value=fill_value)
            data = None
            phi_mhd = None
            th_mhd = None
            r_mhd = None
            return cp.asnumpy(f1_interp((cp.array(phi), cp.array(theta), cp.array(r))))
        else:
            # log_memory_usage("+++++++++ Interp start")
            # breakpoint()
            f1_interp = rgi((phi_mhd, th_mhd, r_mhd), data, method=method, bounds_error=False, fill_value=fill_value)
            # h5file.close()
            del data, phi_mhd, th_mhd, r_mhd
            gc.collect()
            # data = None
            # phi_mhd = None
            # th_mhd = None
            # r_mhd = None
            # log_memory_usage("+++++++++ Interp end")
            return f1_interp((phi, theta, r))

    def forward(self, query_points):
        """Translates x,y,z position and time into a density and temperature map.
            See (Pascoe et al. 2019) https://iopscience.iop.org/article/10.3847/1538-4357/ab3e39

        Args:
            batch (float,vector): Batch of points to render
        Params:
            h0 (float, optional): isothermal scale height. Defaults to 60*u.Mm.
            t0 (_type_, optional): coronal temperature. Defaults to 1.2*u.MK.
            R_s (_type_, optional): Effective solar surface radius. Defaults to 1.2*u.solRad.
            t_photosphere (float, optional): Temperature at the solar surface. Defaults to 5777*u.K.
            rho_0 (float): Density at the solar surface. Defaults to 2e8/u.cm**3.
        Returns:
            rho (float): Density at x,y,z,t
            temp (float): Temperature at x,y,z,t
        """

        # user-chosen coordinates (x,y,z) and time (t) for density and temperature to query
        x = query_points[:, 0]
        y = query_points[:, 1]
        z = query_points[:, 2]
        t = query_points[:, 3]

        # Convert to spherical coordinates
        r = torch.sqrt(x**2 + y**2 + z**2)
        th = torch.arccos(z/r)
        phi = torch.arctan2(y, x)
        phi[phi < 0] += 2*np.pi

        # Initialize output tensors filled with zeros with shape x
        log_ne = torch.zeros_like(x)
        mean_log_T = torch.zeros_like(x)
        # fill_value_density = 1.e-5
        # fill_value_temperature = 1.e-3
        interp_type = 'linear'

        # loop over only unique times
        for time in torch.unique(t):

            # Define logical mask to make interpolation
            mask = (t == time).cpu().numpy()
            r_mask = r[mask].cpu().numpy()
            th_mask = th[mask].cpu().numpy()
            phi_mask = phi[mask].cpu().numpy()

            # Interpolation of frames
            f1 = time * (self.flast - self.ffirst) + self.ffirst
            frame_fraction = f1 - int(f1)
            f2 = torch.ceil(f1).type(torch.int)
            f1 = torch.floor(f1).type(torch.int)

            # Interpolation of density and temperature
            f1_rho = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f1, 'rho',
                                  method=interp_type, fill_value=None)).to(t.device)
            f1_t = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f1, 't',
                                method=interp_type, fill_value=None)).to(t.device)
            f2_rho = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f2, 'rho',
                                  method=interp_type, fill_value=None)).to(t.device)
            f2_t = torch.Tensor(self.interp(phi_mask, th_mask, r_mask, f2, 't',
                                method=interp_type, fill_value=None)).to(t.device)

            # Linear time interpolation of density and temperature
            ne = 1e8*((1-frame_fraction)*f1_rho + frame_fraction*f2_rho) # cm^3
            log_ne[mask] = torch.log10(ne)  
            total_ne = ne.sum(-1)[..., None]
            total_log_ne = torch.log10(total_ne)
            mean_log_T[mask] = torch.log10(2.807066716734894e7*((1-frame_fraction)*f1_t + frame_fraction*f2_t))
            f1_t = None
            f2_t = None
            f1_rho = None
            f2_rho = None
            mask = None
            r_mask = None
            th_mask = None
            phi_mask = None

        # Output density, temperature, absorption and volumetric constant
        return {'log_ne': log_ne, 'log_T': self.log_T,
                'mean_log_T': mean_log_T,
                'total_ne': total_ne,
                'total_log_ne': total_log_ne,
                'ne': ne
                }
