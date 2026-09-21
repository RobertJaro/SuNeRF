import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header

from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from sunerf.train.correction import AlignmentModule


def test_arc_coordinate_helper_preserves_yx_shape_and_uses_wcs_inverse():
    data = np.zeros((3, 5), dtype=np.float32)
    observer = SkyCoord(
        0 * u.deg,
        0 * u.deg,
        1 * u.AU,
        frame=frames.HeliographicStonyhurst,
        obstime='2025-01-01',
    )
    reference = SkyCoord(
        Tx=0 * u.arcsec,
        Ty=0 * u.arcsec,
        observer=observer,
        obstime=observer.obstime,
        frame=frames.Helioprojective,
    )
    header = make_fitswcs_header(data, reference, scale=[1200, 800] * u.arcsec / u.pix)
    header['CTYPE1'] = 'HPLN-ARC'
    header['CTYPE2'] = 'HPLT-ARC'
    s_map = Map(data, header)

    result = get_azimuthal_equidistant_coordinates(s_map)
    expected = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)

    assert result.shape == (3, 5, 2)
    np.testing.assert_allclose(result[..., 0].to_value(u.arcsec), expected.Tx.to_value(u.arcsec))
    np.testing.assert_allclose(result[..., 1].to_value(u.arcsec), expected.Ty.to_value(u.arcsec))


def test_enabled_alignment_has_a_trainable_nonzero_rotation():
    module = AlignmentModule(angle_scale=1e-2)
    rays = torch.tensor([
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ])
    time = torch.tensor([[0.0], [1.0]])

    aligned = module(rays, time)
    aligned[..., 1, 1].sum().backward()

    assert not torch.equal(aligned[..., 1, :], rays[..., 1, :])
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in module.parameters()
    )
