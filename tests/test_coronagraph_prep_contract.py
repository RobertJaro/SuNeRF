import argparse
import ast
import datetime as dt
import inspect
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from sunerf.data.coronagraph import prep_punch_pam, prep_stereo_cor
from sunerf.data.coronagraph.prep_ccor import CCORPrep
from sunerf.data.coronagraph.prep_common import (
    MapPreprocessor,
    _hpc_circle_mask,
    add_common_prep_arguments,
    common_kwargs_from_args,
    select_items_by_time,
)
from sunerf.data.coronagraph.prep_coronagraph import CoronagraphPrep
from sunerf.data.coronagraph.prep_lasco import LascoClearPrep, LascoPrep
from sunerf.data.coronagraph.prep_punch import PunchPrep
from sunerf.data.coronagraph.prep_punch_cam import PunchCamPrep
from sunerf.data.coronagraph.prep_punch_pam import PunchPamPrep
from sunerf.data.coronagraph.prep_punch_triplets import PunchTripletPrep
from sunerf.data.coronagraph.prep_stereo_cor import (
    StereoCorPrep,
    invalid_pixel_fraction,
)


PREP_CLASSES = (
    CCORPrep,
    CoronagraphPrep,
    LascoPrep,
    LascoClearPrep,
    PunchPrep,
    PunchCamPrep,
    PunchPamPrep,
    PunchTripletPrep,
    StereoCorPrep,
)


def test_punch_pam_masks_nonpositive_b_pb_and_clips_nonpositive_pbp(monkeypatch):
    science = np.array(
        [
            [[1.0, 2.0], [-3.0, 4.0]],
            [[-1.0, 2.0], [5.0, 6.0]],
            [[7.0, -8.0], [9.0, 10.0]],
        ]
    )
    packed_uncertainty = np.array(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ]
    )

    def getdata(_path, extension):
        return science if extension == 1 else packed_uncertainty

    monkeypatch.setattr(prep_punch_pam.fits, "getdata", getdata)
    monkeypatch.setattr(prep_punch_pam.fits, "getheader", lambda *_args: {})
    monkeypatch.setattr(
        prep_punch_pam,
        "Map",
        lambda data, _header: type("MapLike", (), {"data": data})(),
    )

    tb_map, pb_map = PunchPamPrep._load_punch_pam_maps("pam.fits")

    np.testing.assert_array_equal(
        tb_map.data,
        np.array([[1.0, 2.0], [np.nan, np.nan]]),
    )
    np.testing.assert_array_equal(
        pb_map.data,
        # pB' = -8 is clipped to zero, so that pixel keeps pB = 2.
        np.array([[np.nan, 2.0], [np.sqrt(106.0), np.nan]]),
    )


@pytest.mark.parametrize("flag", ["OUTLIER", "BADPKTS"])
@pytest.mark.parametrize("flag_value", [-1, 1, 2])
def test_punch_pam_discards_flagged_file_before_loading_data(
    tmp_path,
    monkeypatch,
    flag,
    flag_value,
):
    prepper = PunchPamPrep(tmp_path, remove_solar_system_objects=False)
    monkeypatch.setattr(
        prep_punch_pam.fits,
        "getheader",
        lambda *_args: {"OUTLIER": 0, "BADPKTS": 0, flag: flag_value},
    )

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("flagged data should not be loaded")

    monkeypatch.setattr(prep_punch_pam.fits, "getdata", fail_if_loaded)

    assert prepper.convert("flagged.fits") is None


@pytest.mark.parametrize("prep_class", PREP_CLASSES)
def test_prep_overwrite_is_opt_in(prep_class):
    assert inspect.signature(prep_class).parameters["overwrite"].default is False


def test_common_preprocessor_preserves_nonpositive_values_by_default():
    source_map = type("MapLike", (), {"data": np.array([[-1.0, 0.0, 1.0]])})()

    result = MapPreprocessor(remove_solar_system_objects=False).prepare_map(source_map)

    np.testing.assert_array_equal(result.data, source_map.data)


def test_common_preprocessor_masks_only_explicit_value_limits():
    source_map = type(
        "MapLike",
        (),
        {"data": np.array([[-1.0, 0.0, 1.0, 3.0]])},
    )()

    result = MapPreprocessor(
        remove_solar_system_objects=False,
        value_min=0.0,
        value_max=2.0,
    ).prepare_map(source_map)

    np.testing.assert_array_equal(
        result.data,
        np.array([[np.nan, 0.0, 1.0, np.nan]]),
    )


def test_cor2_invalid_fraction_counts_nonpositive_values_without_modifying_data():
    data = np.array([[-2.0, 0.0, 1.0, np.nan]])
    original = data.copy()

    fraction = invalid_pixel_fraction(data)

    assert fraction == 0.75
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    ("tb_program", "pb_program", "rejected_channel"),
    [
        ("DOUBLE", "NORMAL", "tB"),
        ("NORMAL", "DOUBLE", "pB"),
        (None, "NORMAL", "tB"),
    ],
)
def test_cor2_program_filter_is_enforced_before_existing_output_skip(
    tmp_path,
    monkeypatch,
    tb_program,
    pb_program,
    rejected_channel,
):
    def load_map(path):
        program = tb_program if path == "tb.fits" else pb_program
        header = {} if program is None else {"SEB_PROG": program}
        data = np.ones((2, 2))
        return data, header, object()

    monkeypatch.setattr(prep_stereo_cor, "load_stereo_map", load_map)
    prepper = StereoCorPrep(
        tmp_path,
        overwrite=False,
        remove_solar_system_objects=False,
    )
    (tmp_path / "tB" / "tb.fits").touch()
    (tmp_path / "pB" / "pb.fits").touch()

    result = prepper.convert(("tb.fits", "pb.fits"))

    assert result["status"] == "rejected"
    assert f"{rejected_channel} SEB_PROG=" in result["reason"]


def test_cor2_no_overwrite_preserves_existing_pair_member(tmp_path, monkeypatch):
    saves = []

    class FakeMap:
        def save(self, path, overwrite):
            saves.append((path, overwrite))

    def load_map(_path):
        return np.ones((2, 2)), {"SEB_PROG": "NORMAL"}, FakeMap()

    monkeypatch.setattr(prep_stereo_cor, "load_stereo_map", load_map)
    prepper = StereoCorPrep(
        tmp_path,
        remove_solar_system_objects=False,
    )
    prepper.map_preprocessor.prepare_map = lambda s_map: s_map
    tb_out = tmp_path / "tB" / "tb.fits"
    pb_out = tmp_path / "pB" / "pb.fits"
    tb_out.write_text("preserve me")

    result = prepper.convert(("tb.fits", "pb.fits"))

    assert result["status"] == "processed"
    assert tb_out.read_text() == "preserve me"
    assert saves == [(str(pb_out), False)]


def test_object_mask_uses_circular_hpc_radius():
    y, x = np.indices((5, 5))
    coordinates = type(
        "Coordinates",
        (),
        {"Tx": x * u.arcsec, "Ty": y * u.arcsec},
    )()
    center = type(
        "Coordinate",
        (),
        {"Tx": 2.0 * u.arcsec, "Ty": 2.0 * u.arcsec},
    )()

    mask = _hpc_circle_mask(coordinates, center, radius=1.0)

    assert mask[2, 2]
    assert mask[2, 3]
    assert not mask[2, 4]


def test_time_selection_applies_range_before_cadence():
    base = dt.datetime(2025, 9, 1)
    items = ["a", "b", "c", "d"]
    times = {
        item: base + dt.timedelta(hours=12 * index)
        for index, item in enumerate(items)
    }

    selected = select_items_by_time(
        items,
        start=base + dt.timedelta(hours=6),
        end=base + dt.timedelta(days=2),
        cadence=dt.timedelta(days=1),
        get_time=times.__getitem__,
    )

    assert selected == ["b", "d"]


def get_all_common_kwargs():
    parser = argparse.ArgumentParser()
    add_common_prep_arguments(
        parser,
        include_max_radius=True,
    )
    return common_kwargs_from_args(parser.parse_args([]))


@pytest.mark.parametrize("prep_class", PREP_CLASSES)
def test_prep_constructor_accepts_and_forwards_all_common_kwargs(prep_class, tmp_path):
    common_kwargs = get_all_common_kwargs()
    common_names = set(common_kwargs)
    signature = inspect.signature(prep_class)
    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    missing = common_names - set(signature.parameters)

    assert accepts_arbitrary_kwargs or not missing, (
        f"{prep_class.__name__} does not accept shared preprocessing arguments: "
        f"{sorted(missing)}"
    )

    prepper = prep_class(tmp_path / prep_class.__name__, **common_kwargs)
    for name, expected in common_kwargs.items():
        actual = getattr(prepper.map_preprocessor, name)
        assert actual == expected, (
            f"{prep_class.__name__} did not forward {name}: "
            f"expected {expected!r}, got {actual!r}"
        )


@pytest.mark.parametrize("prep_class", PREP_CLASSES)
def test_prep_call_does_not_duplicate_common_kwargs(prep_class):
    common_names = set(get_all_common_kwargs())
    module = inspect.getmodule(prep_class)
    tree = ast.parse(inspect.getsource(module))
    duplicate_names = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != prep_class.__name__:
            continue
        if not any(keyword.arg is None for keyword in node.keywords):
            continue
        duplicate_names.update(
            keyword.arg
            for keyword in node.keywords
            if keyword.arg in common_names
        )

    assert not duplicate_names, (
        f"{prep_class.__name__} passes shared arguments both explicitly and via "
        f"**common_kwargs_from_args(): {sorted(duplicate_names)}"
    )


def test_all_common_kwargs_consumers_are_covered():
    package_dir = Path(inspect.getfile(CCORPrep)).parent
    detected_consumers = set()

    for module_path in package_dir.glob("prep_*.py"):
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            uses_common_kwargs = any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Call)
                and isinstance(keyword.value.func, ast.Name)
                and keyword.value.func.id == "common_kwargs_from_args"
                for keyword in node.keywords
            )
            if uses_common_kwargs:
                detected_consumers.add((module_path.stem, node.func.id))

    covered_consumers = {
        (prep_class.__module__.rsplit(".", 1)[-1], prep_class.__name__)
        for prep_class in PREP_CLASSES
    }
    assert detected_consumers == covered_consumers
