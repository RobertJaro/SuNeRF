import struct
from datetime import datetime, timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sunpy.map")

from sunerf.data.psi.density_cube import (  # noqa: E402
    PSIDensityCube,
    _read_plain_hdf4_sds,
    dump_index,
)
from sunerf.data.psi.prepare_psi_clear import radial_rsun  # noqa: E402
from sunerf.data.psi.render_psi_thomson import (  # noqa: E402
    frame_schedule,
    line_of_sight_samples,
    observation_map,
    observer_coordinate,
    render_rays,
)
from sunerf.physics.thomson import LIMB_DARKENING_COEFF, R_SUN_CM, SIGMA_NE  # noqa: E402
from sunerf.rendering.thomson import ThomsonScattering  # noqa: E402


def _mesh():
    # MAS meshes carry ghost cells beyond the periodic seam and the poles.
    phi = np.linspace(-0.05, 2 * np.pi + 0.05, 40)
    theta = np.linspace(-0.02, np.pi + 0.02, 30)
    radius = np.geomspace(1.0, 30.0, 50)
    return radius, theta, phi


def _write_plain_hdf4(path, density, radius, theta, phi):
    elements = [
        (702, density.astype(">f4").tobytes()),
        (106, b"\x01\x05\x20\x01"),
        (701, struct.pack(">H3I", 3, *density.shape) + b"\x00" * 16),
        (703, b"\x01\x01\x01" + b"".join(axis.astype(">f4").tobytes() for axis in (phi, theta, radius))),
    ]
    offset = 4 + 6 + 12 * len(elements)
    descriptors = b""
    for tag, payload in elements:
        descriptors += struct.pack(">HHII", tag, 2, offset, len(payload))
        offset += len(payload)
    path.write_bytes(
        b"\x0e\x03\x13\x01" + struct.pack(">HI", len(elements), 0) + descriptors
        + b"".join(payload for _, payload in elements)
    )


def test_plain_hdf4_reader_round_trips_density_and_axes(tmp_path):
    radius, theta, phi = _mesh()
    density = np.random.default_rng(0).random((phi.size, theta.size, radius.size)).astype(np.float32)
    path = tmp_path / "rho000007.hdf"
    _write_plain_hdf4(path, density, radius, theta, phi)

    read_density, read_radius, read_theta, read_phi = _read_plain_hdf4_sds(path)

    np.testing.assert_array_equal(read_density, density)
    np.testing.assert_allclose(read_radius, radius, rtol=1e-6)
    np.testing.assert_allclose(read_theta, theta, rtol=1e-6)
    np.testing.assert_allclose(read_phi, phi, rtol=1e-6)
    assert dump_index(path) == 7


def test_cube_interpolation_is_periodic_and_vanishes_outside_the_mesh():
    radius, theta, phi = _mesh()
    # log n_e is linear in radius and independent of angle apart from a cos(phi) term.
    density = np.exp(-0.1 * radius)[None, None, :] * (2.0 + np.cos(phi))[:, None, None]
    density = np.broadcast_to(density, (phi.size, theta.size, radius.size))
    cube = PSIDensityCube(density, radius, theta, phi, density_unit_scale_cm3=1.0)

    query_radius = torch.tensor([2.5, 2.5, 2.5, 0.5, 31.0])
    query_theta = torch.full((5,), 1.0)
    query_phi = torch.tensor([0.01, 0.01 + 2 * np.pi, 0.01 - 2 * np.pi, 1.0, 1.0])
    values = cube(query_radius, query_theta, query_phi).numpy()

    np.testing.assert_allclose(values[:3], values[0], rtol=1e-5)
    np.testing.assert_allclose(values[0], np.exp(-0.25) * (2.0 + np.cos(0.01)), rtol=5e-3)
    assert values[3] == 0 and values[4] == 0


def test_frame_schedule_places_every_dump_at_its_simulation_time():
    from astropy.time import Time

    start = datetime(2021, 10, 28, 15, 30)
    # The simulation times are irregular; the schedule must not assume a cadence.
    dump_times = {
        1: Time(start),
        2: Time(start + timedelta(minutes=5)),
        50: Time(start + timedelta(minutes=245)),
        51: Time(start + timedelta(minutes=260)),
    }
    schedule = frame_schedule(
        [50, 1, 2], dump_times,
        pre_duration=timedelta(hours=1), pre_cadence=timedelta(minutes=30),
    )

    assert schedule == [
        (start - timedelta(minutes=60), 1),
        (start - timedelta(minutes=30), 1),
        (start, 1),
        (start + timedelta(minutes=5), 2),
        (start + timedelta(minutes=245), 50),
    ]
    assert frame_schedule([1, 2], dump_times) == [
        (start, 1), (start + timedelta(minutes=5), 2),
    ]
    with pytest.raises(ValueError):
        frame_schedule([1], dump_times, pre_duration=timedelta(hours=1))
    with pytest.raises(KeyError):
        frame_schedule([1, 3], dump_times)
    with pytest.raises(ValueError):
        frame_schedule([1, 2], {1: dump_times[2], 2: dump_times[1]})


def test_psi_header_with_tabs_yields_the_dump_time():
    from sunerf.data.psi.dump_times import cadence_summary, dump_time, parse_psi_header

    cards = [
        "SIMPLE  =                    T",
        "DATE_OBS= '\t2021-10-28T19:35:00'",
        "SIM_DUMP=                   50",
        "END",
    ]
    raw = "".join(card.ljust(80) for card in cards).ljust(2880).encode("latin-1")
    assert dump_time(parse_psi_header(raw)) == (50, "2021-10-28T19:35:00.000")

    summary = cadence_summary({1: "2021-10-28T15:30:00", 2: "2021-10-28T15:35:00",
                               4: "2021-10-28T15:45:00"})
    assert summary["cadence_seconds_min"] == summary["cadence_seconds_max"] == 300.0


def test_observer_is_a_lagrange_point_or_an_explicit_stonyhurst_position():
    time = datetime(2021, 10, 28, 15, 30)
    l1 = observer_coordinate(["L1"], time)
    l5 = observer_coordinate(["l5"], time)
    explicit = observer_coordinate(["-30", "10", "0.5"], time)

    assert abs(l1.lon.deg) < 1e-6 and l1.radius.to_value("AU") == pytest.approx(0.99 * l5.radius.to_value("AU"))
    assert l5.lon.deg == pytest.approx(-60.0, abs=1.0)
    assert (explicit.lon.deg, explicit.lat.deg) == pytest.approx((-30.0, 10.0))
    assert explicit.radius.to_value("AU") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        observer_coordinate(["L9"], time)


def test_observation_map_spans_the_requested_field_of_view():
    time = datetime(2021, 10, 28, 15, 30)
    s_map = observation_map(observer_coordinate(["L4"], time), time, outer_rsun=15.0, resolution=64, key="L4")
    radius = radial_rsun(s_map)

    assert s_map.data.shape == (64, 64)
    assert s_map.date.to_datetime() == time
    # The field edge lies half a pixel beyond the outermost pixel centres.
    assert radius[32, 0] == pytest.approx(15.0 * 63 / 64, rel=1e-2)
    assert radius.min() < 0.4


def test_line_of_sight_samples_span_the_chord_through_the_sphere():
    rays_o = np.array([[215.0, 0.0, 0.0]])
    rays_d = np.array([[-np.cos(0.05), np.sin(0.05), 0.0]])
    z = line_of_sight_samples(rays_o, rays_d, sphere_radius=30.0, n_samples=65)[0]
    points = rays_o + z[:, None] * rays_d
    np.testing.assert_allclose(np.linalg.norm(points[[0, -1]], axis=-1), 30.0, rtol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(points[32]), 215.0 * np.sin(0.05), rtol=1e-9)


def _reference_brightness(impact, observer_distance, density_at_1rsun, sphere_radius):
    """Textbook van de Hulst integration for n_e = n_0 r^-2 in mean solar brightness."""
    closest = np.sqrt(observer_distance ** 2 - impact ** 2)
    half = np.sqrt(sphere_radius ** 2 - impact ** 2)
    z = np.linspace(-half, half, 200001)
    r = np.hypot(impact, z)
    so = 1.0 / r
    co = np.sqrt(1.0 - so ** 2)
    log_term = np.log((1.0 + so) / co)
    a = co * so ** 2
    b = -(1.0 - 3.0 * so ** 2 - co ** 2 / so * (1.0 + 3.0 * so ** 2) * log_term) / 8.0
    c = 4.0 / 3.0 - co - co ** 3 / 3.0
    d = (5.0 + so ** 2 - co ** 2 / so * (5.0 - so ** 2) * log_term) / 8.0
    u = LIMB_DARKENING_COEFF
    # The scattering angle obeys sin(chi) = b_sun / r with b_sun the distance of
    # the line of sight from Sun centre.
    polarized = (impact / r) ** 2 * ((1.0 - u) * a + u * b)
    total = 2.0 * ((1.0 - u) * c + u * d) - polarized
    factor = np.pi * SIGMA_NE / 2.0 / (1.0 - u / 3.0) * R_SUN_CM * density_at_1rsun
    assert closest > half
    return (
        factor * np.trapezoid(total / r ** 2, z),
        factor * np.trapezoid(polarized / r ** 2, z),
    )


def test_rendered_brightness_matches_independent_thomson_integration():
    radius, theta, phi = _mesh()
    radius = np.geomspace(1.0, 30.0, 400)
    density_at_1rsun = 1.0e8
    density = np.broadcast_to(radius[None, None, :] ** -2.0, (phi.size, theta.size, radius.size))
    cube = PSIDensityCube(density, radius, theta, phi, density_unit_scale_cm3=density_at_1rsun)

    observer_distance, elongation = 215.0, np.deg2rad(2.0)
    rays = np.array([[
        [observer_distance, 0.0, 0.0],
        [-np.cos(elongation), 0.0, np.sin(elongation)],
    ]])
    image = render_rays(
        cube, ThomsonScattering(Rs_per_ds=1.0), rays, longitude_offset=0.3,
        n_samples=2048, ray_batch=8, device=torch.device("cpu"),
    )[0]

    expected = _reference_brightness(
        observer_distance * np.sin(elongation), observer_distance, density_at_1rsun, 30.0
    )
    np.testing.assert_allclose(image, expected, rtol=5e-3)
    assert image[1] < image[0]
