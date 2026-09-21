from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from sunpy.map import Map

from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.model.thomson import save_thomson_sunerf
from sunerf.train.correction import CorrectionModule


class _TestRendering(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)
        self.rendering_modules = nn.ModuleDict({"L4": nn.Identity()})


class _ConstantCorrection(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, coordinates):
        return self.values.expand(*coordinates.shape[:-1], -1)


def _reference_map():
    header = {
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "cdelt1": 10.0,
        "cdelt2": 10.0,
        "crpix1": 2.0,
        "crpix2": 2.0,
        "crval1": 0.0,
        "crval2": 0.0,
        "date-obs": "2021-10-28T15:30:00",
        "hgln_obs": 0.0,
        "hglt_obs": 0.0,
        "dsun_obs": 149_597_870_691.0,
        "rsun_ref": 696_000_000.0,
        "rsun_obs": 959.63,
    }
    return Map(np.ones((3, 3), dtype=np.float32), header)


def test_thomson_artifact_save_is_atomic_and_supports_filename_only(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    model = SimpleNamespace(
        rendering=nn.Linear(1, 1),
        correction_modules=nn.ModuleDict(),
        calibration_modules=nn.ModuleDict(),
    )
    data_module = SimpleNamespace(
        config={"instrument": {}},
        Rs_per_ds=1.0,
        seconds_per_dt=1.0,
        ref_date=None,
        drho_cm3=2.0,
    )

    save_thomson_sunerf(model, data_module, "state.snf", msb_norm=3.0)

    saved = torch.load(tmp_path / "state.snf", weights_only=False)
    assert saved["thomson_normalization"]["msb_norm"] == 3.0
    assert not list(tmp_path.glob("state.snf.tmp-*"))


def test_learned_degradation_round_trips_through_thomson_loader(tmp_path):
    ref_map = _reference_map()
    correction = CorrectionModule(
        corrections=["tB_add", "pB_add", "tB_mul", "pB_mul"],
    )
    with torch.no_grad():
        for parameter in correction.parameters():
            parameter.fill_(0.125)
    original_state = {
        key: value.detach().clone() for key, value in correction.state_dict().items()
    }
    model = SimpleNamespace(
        rendering=_TestRendering(),
        correction_modules=nn.ModuleDict({"L4": correction}),
        calibration_modules=nn.ModuleDict(),
    )
    data_module = SimpleNamespace(
        config={
            "L4": {
                "type": "thomson",
                "instrument_key": "L4",
                "image_shape": ref_map.data.shape,
                "wcs": ref_map.wcs,
                "times": np.array([ref_map.date.datetime]),
                "image_norm": 512.0,
                "hpc_norm": 1.0e4,
            }
        },
        Rs_per_ds=15.0,
        seconds_per_dt=86400.0,
        ref_date=ref_map.date.datetime,
        drho_cm3=2.0,
    )
    artifact = tmp_path / "degraded.snf"

    save_thomson_sunerf(
        model, data_module, artifact, msb_norm=1.0e-9, msb=1.0, sigma_ne=1.0
    )
    loader = ThomsonSuNeRFLoader(artifact, device="cpu", trusted=True)

    assert set(loader.correction_modules) == {"L4"}
    loaded_state = loader.correction_modules["L4"].state_dict()
    assert loaded_state.keys() == original_state.keys()
    for key in original_state:
        torch.testing.assert_close(loaded_state[key], original_state[key])

    learned = loader.load_correction_masks(ref_map, instrument_key="L4")
    assert set(learned) == {"tB_add", "pB_add", "tB_mul", "pB_mul"}
    for correction_map in learned.values():
        assert np.isfinite(correction_map.data).all()


def test_radial_gain_is_applied_after_channel_additive_terms():
    correction = CorrectionModule(
        corrections=["calibration_gain", "tB_add", "pB_add"],
        additive_scale=1.0,
    )
    correction.radial_correction_module = _ConstantCorrection(
        [np.log(2.0) / 0.01]
    )
    correction.img_correction_module = _ConstantCorrection([3.0, 4.0])
    image = torch.tensor([[5.0, 7.0]])
    coordinates = torch.zeros((1, 3))

    corrected, learned = correction(
        image,
        img_coords=coordinates[:, :2],
        hpc_coords=coordinates,
        time=coordinates[:, :1],
    )

    torch.testing.assert_close(corrected, torch.tensor([[16.0, 22.0]]))
    torch.testing.assert_close(learned["calibration_gain"], torch.tensor([[2.0]]))
