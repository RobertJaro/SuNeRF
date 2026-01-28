import torch
from torch import nn

from sunerf.model.model import SirenNet


class CorrectionModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.correction = SirenNet(in_dim=2, out_dim=2, dim=64, n_layers=4, w0_initial=1)

    def forward(self, image, pix_coords):
        correction = self.get_correction(pix_coords)
        tB = image[..., 0:1]
        pB = image[..., 1:2]
        tB = tB + correction['tB_add']
        pB = pB
        corrected = torch.cat([tB, pB], dim=-1)
        return corrected

    def get_correction(self, pix_coords):
        log_correction = self.correction(pix_coords)
        tB_add = torch.exp(log_correction[..., 0:1] - 8)
        mul = torch.exp(log_correction[..., 1:2] * 0.01)
        return {'tB_add': tB_add, }

class CalibrationModule(nn.Module):

    def __init__(self, start_value=0.0, **kwargs):
        super().__init__()
        self.calibration = nn.Parameter(torch.ones(1, dtype=torch.float32) * start_value, requires_grad=True)

    def forward(self, brightness):
        calibrated = brightness * torch.exp(self.calibration)
        return calibrated