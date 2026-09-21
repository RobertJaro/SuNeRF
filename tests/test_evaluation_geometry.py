import numpy as np
from astropy import units as u

from sunerf.evaluation.loader import ThomsonSuNeRFLoader, _get_scale_from_occ_max


def test_tan_scale_places_outer_pixel_at_requested_impact_parameter():
    distance = 215.032 * u.R_sun
    occ_max = 80 * u.R_sun
    resolution = (512, 512) * u.pix

    scale = _get_scale_from_occ_max(occ_max, distance, resolution)
    tangent_plane_radius = (
        scale[0] * ((resolution[0].to_value(u.pix) - 1) / 2) * u.pix
    ).to_value(u.rad)
    sky_angle = np.arctan(tangent_plane_radius)
    impact = distance * np.sin(sky_angle)

    np.testing.assert_allclose(
        impact.to_value(u.R_sun), occ_max.to_value(u.R_sun), rtol=1e-12
    )


def test_tan_scale_respects_numpy_yx_and_wcs_xy_ordering():
    distance = 215.032 * u.R_sun
    occ_max = 80 * u.R_sun
    resolution = (320, 512) * u.pix

    scale = _get_scale_from_occ_max(occ_max, distance, resolution)
    x_radius = (scale[0] * (511 / 2) * u.pix).to_value(u.rad)
    y_radius = (scale[1] * (319 / 2) * u.pix).to_value(u.rad)

    np.testing.assert_allclose(x_radius, y_radius, rtol=1e-12)


def test_column_density_conversion_includes_model_distance_scale():
    loader = ThomsonSuNeRFLoader.__new__(ThomsonSuNeRFLoader)
    loader.drho_cm3 = 2.5
    loader.Rs_per_ds = 4.0

    converted = loader.convert_column_density(np.array([3.0], dtype=np.float32))
    expected = 3.0 * 2.5 * 4.0 * (1 * u.R_sun).to_value(u.cm)

    np.testing.assert_allclose(converted, expected, rtol=1e-7)
