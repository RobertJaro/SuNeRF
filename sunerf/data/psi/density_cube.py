"""Read PSI/MAS HDF4 density cubes and interpolate them at arbitrary points."""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

# The supplied cubes hold dimensionless MAS mass density without composition
# metadata.  SuNeRF's PSI convention assumes fully ionized pure hydrogen.
MAS_RHO_TO_ELECTRON_CM3 = 1.0e8

_HDF4_MAGIC = b"\x0e\x03\x13\x01"
_HDF4_FLOAT32_BIG_ENDIAN = b"\x01\x05\x20\x01"
_DFTAG_NT, _DFTAG_SDD, _DFTAG_SD, _DFTAG_SDS = 106, 701, 702, 703
_DFTAG_SPECIAL_BIT = 0x4000


def dump_index(path) -> int:
    """Return the simulation dump number encoded at the end of a cube name."""
    match = re.search(r"(\d+)$", Path(path).stem)
    if match is None:
        raise ValueError(f"Could not infer a PSI dump number from {path}")
    return int(match.group(1))


def _read_plain_hdf4_sds(path):
    """Read one uncompressed float32 scientific data set without HDF4 libraries.

    MAS writes a single SDS with its three coordinate scales.  Anything else
    (compression, chunking, several data sets) is left to ``psi_io``.
    """
    buffer = Path(path).read_bytes()
    if buffer[:4] != _HDF4_MAGIC:
        raise ValueError(f"{path} is not an HDF4 file.")
    descriptors = {}
    offset = 4
    while offset:
        count, next_offset = struct.unpack(">HI", buffer[offset:offset + 6])
        for index in range(count):
            start = offset + 6 + 12 * index
            tag, _, data_offset, length = struct.unpack(">HHII", buffer[start:start + 12])
            if tag & _DFTAG_SPECIAL_BIT:
                raise ValueError(f"{path} uses special HDF4 elements; install psi-io with pyhdf.")
            if tag in (_DFTAG_NT, _DFTAG_SDD, _DFTAG_SD, _DFTAG_SDS):
                if tag in descriptors:
                    raise ValueError(f"{path} holds several data sets; install psi-io with pyhdf.")
                descriptors[tag] = (data_offset, length)
        offset = next_offset
    if set(descriptors) != {_DFTAG_NT, _DFTAG_SDD, _DFTAG_SD, _DFTAG_SDS}:
        raise ValueError(f"{path} lacks a plain SDS with scales; install psi-io with pyhdf.")

    def element(tag):
        data_offset, length = descriptors[tag]
        return buffer[data_offset:data_offset + length]

    if element(_DFTAG_NT) != _HDF4_FLOAT32_BIG_ENDIAN:
        raise ValueError(f"{path} is not big-endian float32; install psi-io with pyhdf.")
    dimension_record = element(_DFTAG_SDD)
    rank = struct.unpack(">H", dimension_record[:2])[0]
    shape = struct.unpack(f">{rank}I", dimension_record[2:2 + 4 * rank])
    scales = element(_DFTAG_SDS)
    if rank != 3 or not all(scales[:rank]):
        raise ValueError(f"{path} must hold one 3D data set with all coordinate scales.")
    axes = np.split(np.frombuffer(scales[rank:], dtype=">f4"), np.cumsum(shape)[:-1])
    density = np.frombuffer(element(_DFTAG_SD), dtype=">f4").reshape(shape)
    # HDF4 stores the axes in data order: (phi, theta, radius).
    return density, axes[2], axes[1], axes[0]


def read_psi_density(path):
    """Return ``(density[phi, theta, radius], radius, theta, phi)`` in code units."""
    try:
        from psi_io import read_hdf_data
    except ImportError:
        density, radius, theta, phi = _read_plain_hdf4_sds(path)
    else:
        density, radius, theta, phi = read_hdf_data(str(path))
    density = np.asarray(density, dtype=np.float32)
    radius = np.asarray(radius, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    expected_shape = (phi.size, theta.size, radius.size)
    if density.shape != expected_shape:
        raise ValueError(
            f"PSI density shape is {density.shape}; expected (phi, theta, radius) "
            f"= {expected_shape}."
        )
    return density, radius, theta, phi


class PSIDensityCube(nn.Module):
    """Trilinear log-density interpolation on the non-uniform MAS mesh.

    Longitude is periodic, colatitude is clamped to the mesh, and the density
    vanishes outside the radial extent of the simulation.
    """

    def __init__(self, density, radius, theta, phi, density_unit_scale_cm3=MAS_RHO_TO_ELECTRON_CM3):
        super().__init__()
        density = np.asarray(density, dtype=np.float64) * float(density_unit_scale_cm3)
        phi = np.asarray(phi, dtype=np.float64)
        for name, axis in (("radius", radius), ("theta", theta), ("phi", phi)):
            if np.any(np.diff(axis) <= 0):
                raise ValueError(f"PSI {name} axis must be strictly increasing.")
        # Drop MAS ghost cells and close the periodic seam on both sides.
        unique = (phi >= 0.0) & (phi < 2.0 * np.pi)
        phi, density = phi[unique], density[unique]
        phi = np.concatenate([[phi[-1] - 2.0 * np.pi], phi, [phi[0] + 2.0 * np.pi]])
        density = np.concatenate([density[-1:], density, density[:1]], axis=0)
        positive = density > 0
        if not positive.any():
            raise ValueError("PSI density cube holds no positive values.")
        log_density = np.log(np.where(positive, density, density[positive].min()))
        self.register_buffer("log_density", torch.tensor(log_density, dtype=torch.float32))
        self.register_buffer("radius", torch.tensor(radius, dtype=torch.float32))
        self.register_buffer("theta", torch.tensor(theta, dtype=torch.float32))
        self.register_buffer("phi", torch.tensor(phi, dtype=torch.float32))

    @property
    def radial_range(self):
        return float(self.radius[0]), float(self.radius[-1])

    @staticmethod
    def _bracket(axis, coordinates):
        coordinates = coordinates.clamp(axis[0], axis[-1])
        upper = torch.searchsorted(axis, coordinates.contiguous(), right=True).clamp(1, axis.numel() - 1)
        lower = upper - 1
        weight = (coordinates - axis[lower]) / (axis[upper] - axis[lower])
        return lower, upper, weight

    def forward(self, radius, theta, phi):
        """Electron density in ``cm^-3`` at spherical mesh coordinates (radians, R_sun)."""
        inside = (radius >= self.radius[0]) & (radius <= self.radius[-1])
        r0, r1, wr = self._bracket(self.radius, radius)
        t0, t1, wt = self._bracket(self.theta, theta)
        p0, p1, wp = self._bracket(self.phi, torch.remainder(phi, 2.0 * torch.pi))
        value = 0.0
        for p_index, p_weight in ((p0, 1.0 - wp), (p1, wp)):
            for t_index, t_weight in ((t0, 1.0 - wt), (t1, wt)):
                for r_index, r_weight in ((r0, 1.0 - wr), (r1, wr)):
                    value = value + p_weight * t_weight * r_weight * self.log_density[p_index, t_index, r_index]
        return torch.where(inside, torch.exp(value), torch.zeros_like(value))
