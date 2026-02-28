import torch
from torch import nn

from sunerf.model.model import SirenNet, SirenModel



class CorrectionModule(nn.Module):

    def __init__(self, corrections=None,**kwargs):
        super().__init__()
        self.corrections = ['f_corona'] if corrections is None else corrections # use default corrections if none provided
        possible_img_corrections = ['tB_add', 'pB_add', 'tB_mul', 'pB_mul', 'img', 'transmission', 'leakage']
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

        encoding_config = {'type': 'default', 'w0': 30.}

        if len(img_corrections) > 0:
            self.img_correction_module = SirenModel(in_dim=2, out_dim=len(img_corrections), dim=64, n_layers=4, encoding_config=encoding_config)
        else:
            self.img_correction_module = None
        if len(hpc_corrections) > 0:
            # Tx, Ty, distance
            self.hpc_correction_module = SirenModel(in_dim=2, out_dim=len(hpc_corrections), dim=32, n_layers=2, encoding_config=encoding_config)
            self.distance_scaling_module = SirenModel(in_dim=1, out_dim=len(hpc_corrections), dim=16, n_layers=2, encoding_config={'type': 'default', 'w0': 1.})
        else:
            self.hpc_correction_module = None
        if len(radial_corrections) > 0:
            self.radial_correction_module = SirenModel(in_dim=1, out_dim=len(radial_corrections), dim=32, n_layers=2,
                                                       encoding_config={'type': 'default', 'w0': 1.})
        else:
            self.radial_correction_module = None
        if len(temporal_corrections) > 0:
            self.temporal_correction_module = SirenModel(in_dim=1, out_dim=len(temporal_corrections), dim=32, n_layers=2,
                                                         encoding_config={'type': 'default', 'w0': 1.})
        else:
            self.temporal_correction_module = None

        self.img_corrections = img_corrections
        self.hpc_corrections = hpc_corrections
        self.radial_corrections = radial_corrections
        self.temporal_corrections = temporal_corrections

    def forward(self, image, img_coords, hpc_coords, time):
        tB = image[..., 0:1]
        pB = image[..., 1:2]

        radial_coords = torch.norm(hpc_coords[..., :2], dim=-1, keepdim=True)

        # radial_coords = torch.cat([radial_coords, time], dim=-1)
        # img_coords = torch.cat([img_coords, time], dim=-1)
        # hpc_coords = torch.cat([hpc_coords, time], dim=-1)

        corrections = {}
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
                tB = tB * calibration_gain
                pB = pB * calibration_gain
                corrections['calibration_gain'] = calibration_gain
                i += 1
            if 'calibration_offset' in self.radial_corrections:
                calibration_offset = radial_corrections[..., i:i+1]
                tB = tB + calibration_offset
                pB = pB + calibration_offset
                corrections['calibration_offset'] = calibration_offset
                i += 1

        # 3. image-based corrections (additive/multiplicative/transmission)
        if self.img_correction_module is not None:
            img_corrections = self.img_correction_module(img_coords)

            i = 0
            if 'tB_add' in self.img_corrections:
                tB_add = torch.exp(img_corrections[..., i:i+1] - 6)
                tB = tB + tB_add
                corrections['tB_add'] = tB_add
                i += 1
            if 'pB_add' in self.img_corrections:
                pB_add = torch.exp(img_corrections[..., i:i+1] - 6)
                pB = pB + pB_add
                corrections['pB_add'] = pB_add
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

    def forward(self, brightness):
        calibrated = brightness * torch.exp(self.calibration)
        return calibrated

class AlignmentModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.correction = SirenNet(in_dim=1, out_dim=1, dim=8, n_layers=2, w0_initial=1)

    def forward(self, rays, time):
        theta = self.correction(time) * 0 # small angle in radians

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
                 encoding_config=None,
                 scale=1e-3):
        super().__init__()
        encoding_config = {'type': 'default', 'w0': 30.0} if encoding_config is None else encoding_config
        self.model = SirenModel(in_dim=3, out_dim=2, dim=dim, n_layers=n_layers, encoding_config=encoding_config)

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
