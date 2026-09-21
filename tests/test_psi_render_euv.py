from pathlib import Path

import h5py
import numpy as np
import pytest

from sunerf.data.euv.observation import (
    discover_prepared_files,
    get_prepared_euv_adapter,
    load_prepared_map,
)
from sunerf.data.psi import render_euv
from sunerf.response import ResponseArtifact


def _write_cube(root, frame_id=1813, density=1.0):
    radius = np.linspace(0.99, 1.51, 6, dtype=np.float32)
    colatitude = np.linspace(0.0, np.pi, 7, dtype=np.float32)
    longitude = np.linspace(0.0, 2.0 * np.pi, 9, dtype=np.float32)
    shape = (radius.size, colatitude.size, longitude.size)
    fields = {
        "rho": np.full(shape, density, dtype=np.float32),  # 1e8 cm^-3 per unit
        "t": np.full(shape, 1.5e6 / render_euv.MAS_TEMPERATURE_UNIT_K, dtype=np.float32),
    }
    for field, data in fields.items():
        path = Path(root) / field / f"{field}{frame_id:06d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["dim1"], handle["dim2"], handle["dim3"] = radius, colatitude, longitude
            handle["Data"] = data.T


def _response(tmp_path):
    response_path = tmp_path / "response.npz"
    ResponseArtifact(
        channels=("171", "195"),
        log_temperature=np.array([5.0, 6.0, 7.0]),
        log_density=None,
        response=np.full((2, 3), 1.0e-26),
        response_unit="cm5 DN s-1 pix-1",
        emission_measure_convention="ne2",
        provenance={
            "builder": "unit-test",
            "sensitivity_convention": "reference_epoch",
            "measurement_semantics": "per_native_pixel",
            "native_pixel_solid_angle_sr": 5.9e-11,
        },
    ).save(response_path)
    return response_path


def _aia_response(tmp_path):
    path = tmp_path / "aia_response.npz"
    channels = render_euv.INSTRUMENTS["AIA"]["channels"]
    ResponseArtifact(
        channels=channels,
        log_temperature=np.array([5.0, 6.0, 7.0]),
        log_density=None,
        response=np.full((len(channels), 3), 1.0e-26),
        response_unit="cm5 DN s-1 pix-1",
        emission_measure_convention="ne2",
        provenance={
            "builder": "unit-test",
            "sensitivity_convention": "reference_epoch",
            "measurement_semantics": "per_native_pixel",
            "native_pixel_solid_angle_sr": 8.46e-12,
        },
    ).save(path)
    return path


def test_one_call_renders_one_observer_with_one_frame_per_snapshot_pair(tmp_path, monkeypatch):
    monkeypatch.setitem(render_euv.INSTRUMENTS, "EUVI-A", {
        **render_euv.INSTRUMENTS["EUVI-A"], "channels": ("171", "195"),
    })
    _write_cube(tmp_path / "psi", 1813, density=1.0)
    _write_cube(tmp_path / "psi", 1815, density=2.0)
    common = [
        "--psi-data", str(tmp_path / "psi"), "--instrument", "EUVI-A",
        "--response", str(_response(tmp_path)),
        "--reference-date", "2012-08-10T00:00:00", "--cadence-seconds", "3600",
        "--resolution", "16", "--n-samples", "16", "--n-hierarchical-samples", "8",
        "--no-absorption", "--no-light-travel-time", "--device", "cpu",
    ]
    l4, mars = tmp_path / "euvi_a_l4", tmp_path / "anywhere" / "mars_view"

    # One call per observer and instrument; the caller owns the output path.
    render_euv.main([*common, "--location", "L4", "--out-path", str(l4)])
    render_euv.main([*common, "--location", "mars", "--out-path", str(mars)])

    # One overview JPEG per frame shows every rendered channel.
    from PIL import Image

    overviews = sorted(path.name for path in l4.glob("*.jpg"))
    assert overviews == ["20120810T000000_overview.jpg", "20120810T020000_overview.jpg"]
    with Image.open(l4 / overviews[0]) as overview:
        assert overview.format == "JPEG"
        assert overview.width > 1.5 * overview.height  # two channels side by side

    # Two snapshot pairs x two channels, written directly into each out-path.
    for directory in (l4, mars):
        names = sorted(path.name for path in directory.glob("*.prepared.fits"))
        assert names == [
            "20120810T000000_171.prepared.fits", "20120810T000000_195.prepared.fits",
            "20120810T020000_171.prepared.fits", "20120810T020000_195.prepared.fits",
        ]
    files = sorted(str(path) for path in l4.glob("20120810T000000_*.fits"))
    file_dict, _ = discover_prepared_files(files, ["171", "195"])
    maps = [load_prepared_map(file_dict[channel][0]) for channel in ("171", "195")]
    observation = get_prepared_euv_adapter("EUVI").prepare(
        maps, ["171", "195"],
        source_paths=[file_dict[channel][0] for channel in ("171", "195")],
        strict_metadata=True,
    )
    assert all("DN" in unit for unit in observation.measurement_units)
    image = observation.image[0]
    # On-disk ray through a uniform shell: I = G n_e^2 L with L = 0.5 R_sun.
    expected = 1.0e-26 * 1.0e16 * 0.5 * 6.957e10
    assert image[8, 8] == pytest.approx(expected, rel=0.02)
    assert observation.valid_mask[0, 8, 8] and not observation.valid_mask[0, 0, 0]
    assert np.isnan(image[0, 0])

    # The second snapshot is rendered from its own cube (density doubled) and
    # the field of view in solar radii frames Mars' smaller Sun identically.
    later = load_prepared_map(str(mars / "20120810T020000_171.prepared.fits"))
    assert later.data[8, 8] == pytest.approx(4.0 * expected, rel=0.02)
    assert later.meta["psiframe"] == 1815
    assert later.dsun.to_value("AU") > 1.3
    assert later.scale[0] < maps[0].scale[0]

    import json
    record = json.loads((l4 / "observer.json").read_text())
    assert (record["instrument"], record["location"]) == ("EUVI-A", "L4")
    assert record["time_source"]["note"].startswith("PSI HDF5 cubes contain no time")
    assert [entry["frame_id"] for entry in record["observations"]] == [1813, 1815]
    assert 55.0 < record["observations"][0]["stonyhurst_longitude_deg"] < 65.0


def test_detector_timestamps_include_the_sun_observer_light_travel_time(tmp_path):
    import json
    from datetime import datetime

    _write_cube(tmp_path / "psi", 1813)
    options = {
        "instrument": "EUVI-A", "reference_date": "2012-08-10T00:00:00",
        "cadence_seconds": 3600.0, "response": str(_response(tmp_path)),
        "absorption_artifact": None, "resolution": 8, "n_samples": 8,
        "n_hierarchical_samples": 4, "device": "cpu",
    }
    render_euv.INSTRUMENTS["EUVI-A"] = {**render_euv.INSTRUMENTS["EUVI-A"], "channels": ("171", "195")}
    try:
        for location in ("mercury", "mars"):
            render_euv.render_observer(
                tmp_path / "psi", tmp_path / location, location=location, **options
            )
    finally:
        render_euv.INSTRUMENTS["EUVI-A"] = {
            **render_euv.INSTRUMENTS["EUVI-A"], "channels": ("171", "195", "284"),
        }

    delays = {}
    for location in ("mercury", "mars"):
        record = json.loads((tmp_path / location / "observer.json").read_text())
        observation = record["observations"][0]
        assert record["light_travel_time"] is True
        assert observation["snapshot_time"] == "2012-08-10T00:00:00"
        delay = (
            datetime.fromisoformat(observation["time"])
            - datetime.fromisoformat(observation["snapshot_time"])
        ).total_seconds()
        # d / c with 1 AU = 499.005 s.
        assert delay == pytest.approx(observation["distance_au"] * 499.005, rel=1e-3)
        delays[location] = delay
        stamped = load_prepared_map(str(next((tmp_path / location).glob("*_171.prepared.fits"))))
        assert abs((stamped.date.datetime - datetime.fromisoformat(observation["time"])).total_seconds()) < 1e-3
    # The same snapshot reaches Mars several minutes after it reaches Mercury.
    assert delays["mars"] - delays["mercury"] > 300.0


def test_render_observer_is_callable_from_python_and_validates_its_inputs(tmp_path):
    _write_cube(tmp_path / "psi", 1813)
    options = {
        "reference_date": "2012-08-10T00:00:00", "cadence_seconds": 3600.0,
        "response": str(_response(tmp_path)), "absorption_artifact": None,
        "resolution": 8, "n_samples": 8, "n_hierarchical_samples": 4, "device": "cpu",
    }
    with pytest.raises(ValueError, match="instrument must be one of"):
        render_euv.render_observer(
            tmp_path / "psi", tmp_path / "out", instrument="HMI", location="earth", **options
        )
    with pytest.raises(ValueError, match="must be one of"):
        render_euv.render_observer(
            tmp_path / "psi", tmp_path / "out", instrument="AIA", location="pluto", **options
        )
    with pytest.raises(ValueError, match="cadence_seconds must be positive"):
        render_euv.render_observer(
            tmp_path / "psi", tmp_path / "out", instrument="AIA", location="earth",
            **{**options, "cadence_seconds": 0.0},
        )


def test_overviews_use_instrument_colormaps_and_can_be_disabled(tmp_path):
    assert render_euv.channel_colormap("AIA", "A171") == "sdoaia171"
    assert render_euv.channel_colormap("EUVI-B", "284") == "euvi284"
    # A channel without a dedicated SunPy colormap falls back to a neutral one.
    assert render_euv.channel_colormap("EUVI-A", "999") == "gray"

    _write_cube(tmp_path / "psi", 1813)
    render_euv.main([
        "--psi-data", str(tmp_path / "psi"), "--out-path", str(tmp_path / "out"),
        "--instrument", "AIA", "--location", "earth", "--no-overview", "--no-progress",
        "--response", str(_aia_response(tmp_path)),
        "--reference-date", "2012-08-10T00:00:00", "--cadence-seconds", "3600",
        "--resolution", "8", "--n-samples", "8", "--n-hierarchical-samples", "4",
        "--no-absorption", "--device", "cpu",
    ])
    assert not list((tmp_path / "out").glob("*.jpg"))
    assert len(list((tmp_path / "out").glob("*.prepared.fits"))) == 6


def test_observer_keys_cover_lagrange_points_planets_and_explicit_coordinates():
    from datetime import datetime

    time = datetime.fromisoformat("2012-08-10T00:00:00")
    earth = render_euv.observer_coordinate("earth", time)
    for key in render_euv.OBSERVER_KEYS:
        coordinate = render_euv.observer_coordinate(key, time)
        assert np.isfinite(coordinate.radius.to_value("AU"))
    l1, l2, l3, l5 = (render_euv.observer_coordinate(key, time) for key in ("L1", "L2", "l3", "L5"))
    assert l1.radius < earth.radius < l2.radius
    assert abs(abs(l3.lon.wrap_at("180d").to_value("deg")) - 180.0) < 5.0
    assert -65.0 < l5.lon.wrap_at("180d").to_value("deg") < -55.0
    explicit = render_euv.observer_coordinate("90,-20,0.5", time)
    assert (explicit.lon.to_value("deg"), explicit.lat.to_value("deg")) == (90.0, -20.0)

    with pytest.raises(ValueError, match="must be one of"):
        render_euv.observer_coordinate("pluto", time)


def test_psi_data_path_must_contain_rho_and_t_folders(tmp_path):
    (tmp_path / "rho").mkdir()
    with pytest.raises(FileNotFoundError, match="must contain 'rho' and 't' folders"):
        render_euv.pair_psi_files(tmp_path)
