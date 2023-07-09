import numpy as np
import astropy.units as u

import multiprocessing
import glob
import sys, os
from itertools import repeat

import torch

from sunerf.data.utils import loadMap


from sunerf.train.ray_sampling import get_rays
from sunerf.train.sampling import sample_stratified
from sunerf.train.coordinate_transformation import pose_spherical


def simple_density_temperature(x, y, z, t, h0=60*u.Mm, t0=1.2e6*u.K, R_s=1.2*u.solRad, t_photosphere = 5777*u.K, 
                               rho_0 = 2e8/u.cm**3):
    """Translates x,y,z position and time into a density and temperature map.
        See (Pascoe et al. 2019) https://iopscience.iop.org/article/10.3847/1538-4357/ab3e39 

    Args:
        x (float,vector): x-position in solar radii
        y (float,vector): y-position in solar radii
        z (float,vector): z-position in solar radii
        t (_type_): time value
        h0 (float, optional): isothermal scale height. Defaults to 60*u.Mm.
        t0 (_type_, optional): coronal temperature. Defaults to 1.2*u.MK.
        R_s (_type_, optional): Effective solar surface radius. Defaults to 1.2*u.solRad.
        t_photosphere (float, optional): Temperature at the solar surface. Defaults to 5777*u.K.
        rho_0 (float): Density at the solar surface. Defaults to 2e8/u.cm**3.
    """

    # Radius (distance from the center of the sphere) in solar radii
    radius = np.sqrt(x**2 + y**2 + z**2)
    # Initialize density and temperature
    rho = np.zeros(radius.shape)
    temp = np.zeros(radius.shape)
    
    #find indices of radii less than solRad and greater than solRad
    less_than_index = np.where(radius <= 1*u.solRad)
    else_index = np.where(radius > 1*u.solRad)

    # If radius is less then 1 solar radii...
    rho[less_than_index] = rho_0
    # If radius is greater than 1 solar radii...
    rho[else_index] = rho_0 * np.exp(1*u.solRad/h0*(1*u.solRad/radius[else_index]-1)) #See equation 4 in Pascoe et al. 2019

    # Simple temperature model (depends on radius)
    # If radius is less then 1 solar radii...
    temp[less_than_index] = t_photosphere
    # If radius is between 1 solar radii and R_s solar radii...
    R_s_index = np.where(( radius > 1*u.solRad) & (radius <= R_s))
    temp[R_s_index] = (radius[R_s_index]-1*u.solRad)*((t0-t_photosphere)/(R_s - 1*u.solRad))+ t_photosphere #See equation 6 in Pascoe et al. 2019
    # If radius is greater than R_s solar radii, use constant...
    out_sun_index =np.where(radius > R_s)  
    temp[out_sun_index]= t0  

    # Output density and temperature
    return rho, temp

def _retrieve_query_points(imgs_path, resolution=1024):
    aia_paths = sorted(glob.glob(imgs_path+'/*'))
    print('Nb. of files: ', len(aia_paths))


    with multiprocessing.Pool(os.cpu_count()) as p:
            sdo_maps = p.starmap(loadMap, zip(aia_paths, repeat(resolution)))

    views = [((s_map.carrington_latitude.to(u.deg).value, -s_map.carrington_longitude.to(u.deg).value), s_map) for s_map in [*sdo_maps]]

    ref_map = views[0][1]
    scale = ref_map.scale[0] # scale of pixels [arcsec/pixel]
    W = ref_map.data.shape[0] # number of pixels

    distance =  (1 * u.AU).to(u.solRad).value #distance between camera and solar center

    # compute focal length from Helioprojective coordinates (FOV) [pixels]
    focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi/180) 
    focal = np.array(focal, dtype = np.float32)

    # create poses from spherical coordinates
    poses = [pose_spherical(lon, lat, distance) for (lat,lon),s_map in views]
    poses = np.stack(poses).astype(np.float32)

    # create times for maps
    times = [s_map.date.datetime for (lat,lon),s_map in views]
    start_time, end_time = min(times), max(times)
    # times = np.array([(t - start_time) / (end_time - start_time) for t in times], dtype=np.float32)

    data = {'poses':poses, 'focal':focal, 'times': times}

    height = resolution
    width = resolution
    all_rays = torch.stack([torch.stack(get_rays(height, width, focal, p), -2)
                            for p in torch.tensor(poses[0:1])], 0) #shape is [number of poses, height, width, ray_o and ray_d, (x,y,z)]
    flat_rays = all_rays.view((-1, 2, 3))
    rays_o = flat_rays[:, 0]
    rays_d = flat_rays[:, 1]

    # integrate rays from -1.1 to .1 solar radii
        # images processed from ITI to be 2.2 solar radii
    near, far = (1 * u.AU - 1.1 * u.solRad).to(u.solRad).value , (1 * u.AU + .1 * u.solRad).to(u.solRad).value

    n_samples = 64
    perturb = False
    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
    }

    query_points, z_vals = sample_stratified(rays_o, rays_d, near, far, **kwargs_sample_stratified)

    ## TODO: Verifiy the ranges of x, y, z
    x = query_points[:,:,0].numpy()*u.solRad
    y = query_points[:,:,1].numpy()*u.solRad
    z = query_points[:,:,2].numpy()*u.solRad

    return x,y,z


if __name__ == "__main__":
    
    imgs_path = '/mnt/aia-1h/psi-aia/PSI/AIA_193'
    x,y,z = _retrieve_query_points(imgs_path,resolution = 1024)
    density, temp= simple_density_temperature(x,y,z,t=0)
    print(density,temp)

