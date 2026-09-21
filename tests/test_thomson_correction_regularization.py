import math

import pytest
import torch
from torch import nn

from sunerf.model.thomson import ThomsonSuNeRFModule


class _FixedCorrections(nn.Module):
    def __init__(self):
        super().__init__()
        self.additive = nn.Parameter(torch.tensor([[0.02, -0.04], [-0.01, 0.03]]))

    def forward(self, image, image_coords, hpc_coords, time):
        gain = image.new_full((len(image), 1), 1.2)
        return (image + self.additive) * gain, {
            'tB_add': self.additive[:, :1],
            'pB_add': self.additive[:, 1:],
            'calibration_gain': gain,
        }


@pytest.mark.parametrize('instrument,a', [('L4', 1e-5), ('L5', 0.1)])
@pytest.mark.parametrize('mask_channels', [None, 1, 2])
def test_additive_regularization_uses_instrument_image_scaling(instrument, a, mask_channels):
    module = ThomsonSuNeRFModule(
        Rs_per_ds=15,
        seconds_per_dt=86400,
        instruments=[
            {'key': 'L4', 'type': 'default', 'scaling': {'type': 'asinh', 'a': 1e-5}},
            {'key': 'L5', 'type': 'default', 'scaling': {'type': 'asinh', 'a': 0.1}},
        ],
        model_config={'model_type': 'default', 'dim': 8, 'n_layers': 1},
        validation_dataset_mapping={},
        lambda_config={key: 1e-4 for key in (
            'tB_add', 'pB_add', 'tB_add_mean', 'pB_add_mean', 'calibration_gain',
        )},
    )
    correction = _FixedCorrections()
    module.correction_modules[instrument] = correction
    batch = {
        'instrument': instrument,
        'rays': torch.zeros(2, 2, 3),
        'image_coords': torch.zeros(2, 2),
        'hpc_coords': torch.zeros(2, 3),
        'time': torch.zeros(2, 1),
    }
    if mask_channels is not None:
        batch['scaling_mask'] = torch.tensor([[2.0, 4.0], [3.0, 5.0]])[:, :mask_channels]
    image = torch.ones(2, 2)
    losses = []
    corrected, aux = module._apply_image_modules(
        batch, image, correction_losses=losses, collect_regularization=True,
    )

    # Regularization must not transform the forward-model brightness or masks.
    torch.testing.assert_close(corrected, (image + correction.additive) * 1.2)
    torch.testing.assert_close(aux['correction']['tB_add'], correction.additive[:, :1])
    torch.testing.assert_close(aux['correction']['pB_add'], correction.additive[:, 1:])
    torch.testing.assert_close(losses[0]['calibration_gain'], torch.full((2, 1), 0.04))

    mask = batch.get('scaling_mask', 1.0)
    expected_scaled = torch.asinh(correction.additive / mask / a) / math.asinh(1 / a)
    for channel, key in enumerate(('tB_add', 'pB_add')):
        expected = expected_scaled[:, channel:channel + 1]
        torch.testing.assert_close(losses[0][key], expected.square())
        torch.testing.assert_close(losses[0][f'{key}_mean'], expected.mean().square().reshape(1, 1))

    actual_loss = losses[0]['tB_add'].mean() + losses[0]['pB_add'].mean()
    expected_loss = expected_scaled.square().mean(dim=0).sum()
    actual_gradient = torch.autograd.grad(actual_loss, correction.additive, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected_loss, correction.additive)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all()
