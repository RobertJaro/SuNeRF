import torch

from sunerf.model.thomson import ThomsonSuNeRFModule


class _RatioModule:
    ratio_range = (0.0, 3.0)
    ratio_epsilon = 1.0e-6
    _polarization_ratio = ThomsonSuNeRFModule._polarization_ratio
    _polarization_prediction = ThomsonSuNeRFModule._polarization_prediction


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


def test_polarization_prediction_is_linear_in_the_target_and_masks_invalid_model_pixels():
    target = torch.tensor([
        [2.0, 0.6],            # regular pixel
        [0.01, 0.3],           # faint tB: a direct ratio would amplify its noise
        [-0.5, 0.1],           # over-subtracted target tB stays usable
        [1.0, float('nan')],   # missing pB target
    ])
    model = torch.tensor([
        [4.0, 1.2],            # p = 0.3
        [1.0, 0.5],            # p = 0.5
        [1.0, 0.2],            # p = 0.2
        [1.0, 0.5],
    ], requires_grad=True)
    predicted, valid = _RatioModule()._polarization_prediction(target, model)

    assert valid.tolist() == [True, True, True, False]
    torch.testing.assert_close(predicted[:, 0], target[:, 0])
    torch.testing.assert_close(predicted[:3, 1], torch.tensor([0.6, 0.005, -0.1]), rtol=1e-5, atol=1e-6)
    # the regular pixel has the right polarization: zero residual and gradient
    residual = (predicted[valid, 1] - target[valid, 1]).pow(2)
    assert residual[0].item() < 1e-10
    residual.sum().backward()
    assert torch.isfinite(model.grad).all()
    assert torch.all(model.grad[0].abs() < 1e-6) and torch.all(model.grad[3] == 0)


def test_polarization_prediction_gradients_stay_finite_for_non_finite_target_tB():
    # A NaN target tB is masked out of the loss, but the product with the model
    # polarization must not leak NaN into the gradients of the model image.
    model_image = torch.tensor([[1.0, 0.5], [2.0, 0.5]], requires_grad=True)
    target_image = torch.tensor([[float('nan'), 0.4], [2.0, 0.9]])

    predicted, mask = _RatioModule()._polarization_prediction(target_image, model_image)
    (predicted[mask, 1] - target_image[mask, 1]).pow(2).sum().backward()

    assert mask.tolist() == [False, True]
    assert torch.isfinite(model_image.grad).all()
