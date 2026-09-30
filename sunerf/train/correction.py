import numpy as np
import torch
from torch import nn

from sunerf.model.model import SirenNet, SirenModel



class CorrectionModule(nn.Module):

    def __init__(self, corrections=None, additive_scale=1e-4,
                 tB_additive_scale=None, pB_additive_scale=None,
                 additive_parametrization='linear', **kwargs):
        super().__init__()
        # Additive image-plane terms (tB_add, pB_add) map the network output to
        # brightness either linearly, add = scale * raw (one output channel), or
        # as add = scale * a * exp(b) (two channels). The exponential form is
        # linear in the asinh loss space and covers the dynamic range of a
        # coronagraph image from O(1) outputs; the sign is carried by a, so it
        # never has to pass through the flat region of the exponential.
        if additive_parametrization not in ('linear', 'exp'):
            raise ValueError("additive_parametrization must be 'linear' or 'exp'.")
        self.additive_parametrization = additive_parametrization
        self.corrections = ['f_corona'] if corrections is None else corrections # use default corrections if none provided
        self.tB_additive_scale = float(
            additive_scale if tB_additive_scale is None else tB_additive_scale
        )
        self.pB_additive_scale = float(
            additive_scale if pB_additive_scale is None else pB_additive_scale
        )
        if self.tB_additive_scale <= 0 or self.pB_additive_scale <= 0:
            raise ValueError("Additive correction scales must be positive.")
        possible_img_corrections = ['tB_add', 'pB_add', 'tB_straylight', 'pB_straylight', 'tB_mul', 'pB_mul', 'img', 'transmission', 'leakage']
        possible_hpc_corrections = ['f_corona']
        possible_radial_corrections = ['calibration_gain', 'calibration_offset']
        possible_temporal_corrections = ['calibration']
        possible_corrections = possible_img_corrections + possible_hpc_corrections + possible_radial_corrections + possible_temporal_corrections
        for corr in self.corrections:
            if corr not in possible_corrections:
                raise ValueError(f"Unknown correction type: {corr}. Possible types: {possible_corrections}")

        img_corrections = [c for c in self.corrections if c in possible_img_corrections]
        hpc_corrections = [c for c in self.corrections if c in possible_hpc_corrections]
        radial_corrections = [c for c in self.corrections if c in possible_radial_corrections]
        temporal_corrections = [c for c in self.corrections if c in possible_temporal_corrections]

        # additive terms take two output channels in the exponential form
        self.img_channels = {
            c: (2 if c in ('tB_add', 'pB_add') and self.additive_parametrization == 'exp' else 1)
            for c in img_corrections
        }
        if len(img_corrections) > 0:
            # w0=1 resolves only the smooth component of the image plane;
            # w0=30 (standard SIREN) would recover pixel-scale structure.
            self.img_correction_module = SirenModel(in_dim=2, out_dim=sum(self.img_channels.values()), dim=64,
                                                    n_layers=4, w0=1., w0_init=30)
        else:
            self.img_correction_module = None
        if len(hpc_corrections) > 0:
            # Tx, Ty, distance
            self.hpc_correction_module = SirenModel(in_dim=2, out_dim=len(hpc_corrections), dim=32, n_layers=2, w0_init=30)
            self.distance_scaling_module = SirenModel(in_dim=1, out_dim=len(hpc_corrections), dim=16, n_layers=2, w0_init=1)
        else:
            self.hpc_correction_module = None
        if len(radial_corrections) > 0:
            self.radial_correction_module = SirenModel(in_dim=1, out_dim=len(radial_corrections), dim=32, n_layers=2, w0_init=1)
        else:
            self.radial_correction_module = None
        if len(temporal_corrections) > 0:
            self.temporal_correction_module = SirenModel(in_dim=1, out_dim=len(temporal_corrections), dim=32, n_layers=2, w0_init=1)
        else:
            self.temporal_correction_module = None

        self.img_corrections = img_corrections
        self.hpc_corrections = hpc_corrections
        self.radial_corrections = radial_corrections
        self.temporal_corrections = temporal_corrections

    def _additive(self, raw, scale):
        """Map the raw network output (1 or 2 channels) to a signed additive brightness."""
        if self.additive_parametrization == 'exp':
            return scale * raw[..., 0:1] * torch.exp(raw[..., 1:2])
        return raw * scale

    def forward(self, image, img_coords, hpc_coords, time):
        tB = image[..., 0:1]
        pB = image[..., 1:2]

        radial_coords = torch.norm(hpc_coords[..., :2], dim=-1, keepdim=True)

        # radial_coords = torch.cat([radial_coords, time], dim=-1)
        # img_coords = torch.cat([img_coords, time], dim=-1)
        # hpc_coords = torch.cat([hpc_coords, time], dim=-1)

        corrections = {}
        calibration_gain = None
        calibration_offset = None
        # 1. corrections of physical origin (F corona)
        if self.hpc_correction_module is not None:
            hpc_corrections = self.hpc_correction_module(hpc_coords[..., :2])
            distance_scaling = self.distance_scaling_module(hpc_coords[..., 2:3])
            hpc_corrections = hpc_corrections + distance_scaling
            i = 0
            if 'f_corona' in self.hpc_corrections:
                f_corona = torch.exp(hpc_corrections[..., i:i+1] - 6)
                tB = tB + f_corona
                corrections['f_corona'] = f_corona
                i += 1

        # 2. radial corrections (calibration gain/offset)
        if self.radial_correction_module is not None:
            radial_corrections = self.radial_correction_module(radial_coords)
            i = 0
            if 'calibration_gain' in self.radial_corrections:
                calibration_gain = torch.exp(radial_corrections[..., i:i+1] * 0.01)
                corrections['calibration_gain'] = calibration_gain
                i += 1
            if 'calibration_offset' in self.radial_corrections:
                calibration_offset = radial_corrections[..., i:i+1]
                corrections['calibration_offset'] = calibration_offset
                i += 1

        # 3. image-based corrections (additive/multiplicative/transmission)
        if self.img_correction_module is not None:
            img_corrections = self.img_correction_module(img_coords)

            i = 0
            if 'tB_add' in self.img_corrections:
                n = self.img_channels['tB_add']
                tB_add = self._additive(img_corrections[..., i:i + n], self.tB_additive_scale)
                tB = tB + tB_add
                corrections['tB_add'] = tB_add
                i += n
            if 'pB_add' in self.img_corrections:
                n = self.img_channels['pB_add']
                pB_add = self._additive(img_corrections[..., i:i + n], self.pB_additive_scale)
                pB = pB + pB_add
                corrections['pB_add'] = pB_add
                i += n
            if 'tB_straylight' in self.img_corrections:
                tB_straylight = torch.exp(img_corrections[..., i:i+1] - 6.0)
                tB = tB + tB_straylight
                corrections['tB_straylight'] = tB_straylight
                i += 1
            if 'pB_straylight' in self.img_corrections:
                pB_straylight = torch.exp(img_corrections[..., i:i+1] - 6.0)
                pB = pB + pB_straylight
                corrections['pB_straylight'] = pB_straylight
                i += 1
            if 'tB_mul' in self.img_corrections:
                tB_mul = torch.exp(img_corrections[..., i:i+1] * 0.01)
                tB = tB * tB_mul
                corrections['tB_mul'] = tB_mul
                i += 1
            if 'pB_mul' in self.img_corrections:
                pB_mul = torch.exp(img_corrections[..., i:i+1] * 0.01)
                pB = pB * pB_mul
                corrections['pB_mul'] = pB_mul
                i += 1
            if 'transmission' in self.img_corrections:
                transmission = 1 - torch.sigmoid(img_corrections[..., i:i+1] * 0.01)
                tB = tB * transmission
                pB = pB * transmission
                corrections['transmission'] = transmission
                i += 1
            if 'leakage' in self.img_corrections:
                raw = img_corrections[..., i:i + 1]
                leakage = torch.exp(raw - 6.0)

                pB = pB + tB * leakage
                corrections['leakage'] = leakage
                i += 1

        # Apply shared radial calibration after image-coordinate additive terms,
        # matching the synthetic forward model: (image + additive) * gain.
        if calibration_offset is not None:
            tB = tB + calibration_offset
            pB = pB + calibration_offset
        if calibration_gain is not None:
            tB = tB * calibration_gain
            pB = pB * calibration_gain

        # 4. temporal corrections (calibration)
        if self.temporal_correction_module is not None:
            temporal_corrections = self.temporal_correction_module(time)
            i = 0
            if 'calibration' in self.temporal_corrections:
                calibration = torch.exp(temporal_corrections[..., i:i+1] * 0.01)
                tB = tB * calibration
                pB = pB * calibration
                corrections['calibration'] = calibration
                i += 1

        corrected = torch.cat([tB, pB], dim=-1)
        return corrected, corrections

class CalibrationModule(nn.Module):

    def __init__(self, start_value=0.0, **kwargs):
        super().__init__()
        self.calibration = nn.Parameter(torch.ones(1, dtype=torch.float32) * start_value, requires_grad=True)

    def forward(self, value):
        calibrated = value * torch.exp(self.calibration)
        return calibrated

class AlignmentModule(nn.Module):

    def __init__(self, angle_scale=1e-3, **kwargs):
        super().__init__()
        self.angle_scale = float(angle_scale)
        if self.angle_scale <= 0:
            raise ValueError("angle_scale must be positive.")
        self.correction = SirenNet(in_dim=1, out_dim=1, dim=8, n_layers=2, w0_initial=1)

    def forward(self, rays, time):
        # Keep the learned rotation small without severing its gradient.  The old
        # ``* 0`` made an enabled alignment module an exact, untrainable no-op.
        theta = self.correction(time) * self.angle_scale

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        zeroes = torch.zeros_like(cos_theta)
        ones = torch.ones_like(cos_theta)
        rotation_matrix = torch.cat([cos_theta, -sin_theta, zeroes, sin_theta, cos_theta, zeroes, zeroes, zeroes, ones], dim=-1).view(*theta.shape[:-1], 3, 3)

        rays_d = rays[..., 1, :]  # direction vectors
        aligned_rays_d = torch.einsum('...ij,...j->...i', rotation_matrix, rays_d)
        aligned_rays = torch.stack([rays[..., 0, :], aligned_rays_d], dim=-2)  # keep origins the same

        return aligned_rays


class StarBackgroundModule(nn.Module):
    """
    NeRF-style infinite background model: static sky brightness as a function of ray direction only.

    - Input: ray directions (rays_d) in a FIXED inertial/sky frame.
    - Output: additive background in (tB, pB), typically with pB strongly suppressed.

    Call pattern (matching your other correction modules):
        corrected_image, bg_terms = module(image, rays_d)
    """

    def __init__(self,
                 dim=64,
                 n_layers=4,
                 scale=1e-3):
        super().__init__()
        self.model = SirenModel(in_dim=3, out_dim=2, dim=dim, n_layers=n_layers, w0_init=100)

        # Keep background small by construction so it can't trivially explain the corona
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32), requires_grad=False)

    def forward(self, rays_d):
        """
        image: (..., 2) [tB, pB]
        rays_d: (..., 3) ray directions (must be in a fixed inertial/sky frame)
        """
        d = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-8)

        model_out = self.model(d)  # (..., len(outputs))

        tB_add = torch.exp(model_out[..., 0:1]) * self.scale
        pB_add = torch.exp(model_out[..., 1:2]) * self.scale

        correction = torch.cat([tB_add, pB_add], dim=-1)
        return correction


def _rodrigues_rotate(vectors, axis, angle):
    """Rotate ``vectors`` about the unit ``axis`` by ``angle`` (right-handed).

    ``vectors`` and ``axis`` broadcast over leading dimensions with a trailing
    dimension of 3; ``angle`` broadcasts with a trailing dimension of 1.
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    axis_dot = (axis * vectors).sum(dim=-1, keepdim=True)
    return (
        vectors * cos
        + torch.cross(axis, vectors, dim=-1) * sin
        + axis * axis_dot * (1.0 - cos)
    )


class PointingAlignmentModule(nn.Module):
    """Learnable per-instrument pointing correction: image roll plus a shift.

    The correction acts on the ray directions in the world frame and leaves the
    observer positions untouched. Each ray's camera frame is rebuilt from its
    origin exactly like ``pose_spherical``: the boresight points at the Sun
    center, the image ``y`` axis is the projection of solar north (``+z``) and
    ``x`` completes the right-handed frame. The roll rotates the directions
    about the boresight; the shift is a helioprojective pointing offset
    (a rotation of the boresight about the image ``y`` and ``x`` axes), which
    for a distant observer is a rigid translation of the image.

    All three quantities are constant per instrument, so they describe a
    systematic registration error shared by all observations (and apply to
    held-out observations of the same instrument). They are bounded by tanh,
    with ``roll_limit_deg`` and ``shift_limit_arcsec`` as hard limits, and start
    at zero (an exact identity).
    """

    def __init__(self, roll_limit_deg=1.0, shift_limit_arcsec=30.0, trainable=True, **kwargs):
        super().__init__()
        unknown = sorted(kwargs)
        if unknown:
            raise ValueError(f"alignment contains unsupported fields: {unknown}")
        roll_limit_deg = float(roll_limit_deg)
        shift_limit_arcsec = float(shift_limit_arcsec)
        if not (roll_limit_deg > 0 and shift_limit_arcsec > 0):
            raise ValueError("alignment roll_limit_deg and shift_limit_arcsec must be positive.")
        self.roll_limit_deg = roll_limit_deg
        self.shift_limit_arcsec = shift_limit_arcsec
        self.register_buffer('roll_limit_rad', torch.tensor(np.deg2rad(roll_limit_deg), dtype=torch.float32))
        self.register_buffer('shift_limit_rad',
                             torch.tensor(np.deg2rad(shift_limit_arcsec / 3600.0), dtype=torch.float32))
        # Unbounded raw parameters; the effective values are limit * tanh(raw).
        self.raw_roll = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.raw_shift = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.requires_grad_(bool(trainable))

    @property
    def roll_rad(self):
        return self.roll_limit_rad * torch.tanh(self.raw_roll)

    @property
    def shift_rad(self):
        return self.shift_limit_rad * torch.tanh(self.raw_shift)

    @property
    def roll_deg(self):
        return torch.rad2deg(self.roll_rad).reshape(())

    @property
    def shift_arcsec(self):
        return torch.rad2deg(self.shift_rad) * 3600.0

    @staticmethod
    def camera_axes(rays_o):
        """Per-ray (right, up, forward) unit axes matching ``pose_spherical``."""
        forward = -rays_o / (torch.linalg.norm(rays_o, dim=-1, keepdim=True) + 1e-8)
        up0 = torch.zeros_like(forward)
        up0[..., 2] = 1.0
        # Same pole guard as pose_spherical: fall back to +y near the poles.
        polar = (forward[..., 2:3].abs() > 0.999)
        up0_polar = torch.zeros_like(forward)
        up0_polar[..., 1] = 1.0
        up0 = torch.where(polar, up0_polar, up0)
        right = torch.cross(forward, up0, dim=-1)
        right = right / (torch.linalg.norm(right, dim=-1, keepdim=True) + 1e-8)
        up = torch.cross(right, forward, dim=-1)
        return right, up, forward

    def forward(self, rays):
        rays_o = rays[..., 0, :]
        rays_d = rays[..., 1, :]
        right, up, forward = self.camera_axes(rays_o)

        roll = self.roll_rad.to(rays_d.dtype)
        shift = self.shift_rad.to(rays_d.dtype)
        # Roll about the camera z axis (-forward), so a positive roll turns the
        # directions counterclockwise in the (x, y) image plane, then tilt the
        # pointing: a positive x shift rotates the directions about the image y
        # axis, a positive y shift about the image x axis, both with the sign
        # of an increasing helioprojective (Tx, Ty) of the pixel.
        aligned = _rodrigues_rotate(rays_d, -forward, roll)
        aligned = _rodrigues_rotate(aligned, up, -shift[0:1])
        aligned = _rodrigues_rotate(aligned, right, shift[1:2])
        aligned = aligned / (torch.linalg.norm(aligned, dim=-1, keepdim=True) + 1e-8)
        return torch.stack([rays_o, aligned], dim=-2)
