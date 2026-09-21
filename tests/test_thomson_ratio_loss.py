import torch

from sunerf.model.thomson import ThomsonSuNeRFModule


class _RatioModule:
    ratio_range = (0.0, 3.0)
    ratio_epsilon = 1.0e-6
    _polarization_ratio = ThomsonSuNeRFModule._polarization_ratio


def test_polarization_ratio_masks_pixels_outside_the_range():
    image = torch.tensor([
        [1.0, 0.5],  # regular pixel
        [1.0, 2.5],  # unphysical but inside the accepted range
        [1.0, 3.5],  # above the range
        [1.0, -0.1],  # negative ratio
        [-1.0, 0.5],  # non-positive tB
        [float('nan'), 0.5],
        [1.0, float('nan')],
    ])
    ratio, valid = _RatioModule()._polarization_ratio(image)
    assert valid.tolist() == [True, True, False, False, False, False, False]
    torch.testing.assert_close(ratio[:2], torch.tensor([0.5, 2.5]), rtol=1e-5, atol=1e-5)


def test_polarization_ratio_gradients_stay_finite_for_vanishing_model_tB():
    image = torch.tensor([[1.0, 0.5], [-1.0e-6, 0.1], [0.0, 0.0]], requires_grad=True)
    ratio, valid = _RatioModule()._polarization_ratio(image)
    (ratio[valid] - 0.3).pow(2).mean().backward()
    assert valid.tolist() == [True, False, False]
    assert torch.isfinite(image.grad).all()
    assert torch.all(image.grad[1:] == 0)
