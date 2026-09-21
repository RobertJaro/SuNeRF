from pathlib import Path

import numpy as np

from sunerf.response.builder import SpectralEmissivityGrid
from sunerf.response import emissivity


def _grid():
    return SpectralEmissivityGrid(
        wavelength_angstrom=emissivity.WAVELENGTH_ANGSTROM,
        log_temperature=emissivity.LOG_TEMPERATURE,
        log_density=emissivity.LOG_DENSITY,
        emissivity=np.ones(
            (emissivity.LOG_DENSITY.size, emissivity.LOG_TEMPERATURE.size, 991)
        ),
        emissivity_unit="ph cm3 / (Angstrom s sr)",
        emission_measure_convention="ne2",
        provenance={
            "provider": {"name": "fiasco", "version": "0.8.2"},
            "atomic_database": {
                "name": "CHIANTI",
                "version": "11.0.2",
                "sha256": "0" * 64,
            },
            "abundance": {
                "name": "sun_coronal_2021_chianti",
                "version": "11.0.2",
                "sha256": emissivity.ABUNDANCE_SHA256,
            },
            "ionization_equilibrium": {
                "name": "chianti",
                "version": "11.0.2",
                "sha256": emissivity.IONIZATION_EQUILIBRIUM_SHA256,
            },
            "emission_components": ["lines", "free_free", "free_bound", "two_photon"],
        },
    )


def test_generate_reuses_matching_artifact(tmp_path, monkeypatch):
    output = emissivity.default_output_path(tmp_path)
    _grid().save(output)
    monkeypatch.setattr(
        emissivity,
        "prepare_chianti_database",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse")),
    )

    assert emissivity.generate_spectral_emissivity(tmp_path) == output


def test_generate_uses_pinned_grid(tmp_path, monkeypatch):
    atomic_root = tmp_path / "chianti" / "11.0.2"
    sources = {
        "database": atomic_root / "chianti_11.0.2.h5",
        "abundance": atomic_root / "ascii/abundance/sun_coronal_2021_chianti.abund",
        "ionization_equilibrium": atomic_root / "ascii/ioneq/chianti.ioneq",
    }
    captured = {}

    monkeypatch.setattr(
        emissivity,
        "prepare_chianti_database",
        lambda root, **kwargs: captured.update(root=Path(root), prepare=kwargs) or sources,
    )

    class Model:
        def rasterize(self, **kwargs):
            captured["rasterize"] = kwargs
            return _grid()

    def compute(config, log_temperature, log_density, wavelength, **kwargs):
        captured.update(
            config=config,
            log_temperature=log_temperature,
            log_density=log_density,
            wavelength=wavelength,
            compute=kwargs,
        )
        return Model()

    monkeypatch.setattr(emissivity, "compute_fiasco_emission_model", compute)
    output = emissivity.generate_spectral_emissivity(tmp_path, show_progress=False)

    assert output == emissivity.default_output_path(tmp_path)
    assert captured["root"] == atomic_root
    np.testing.assert_array_equal(captured["log_temperature"], emissivity.LOG_TEMPERATURE)
    np.testing.assert_array_equal(captured["log_density"], emissivity.LOG_DENSITY)
    np.testing.assert_array_equal(captured["wavelength"], emissivity.WAVELENGTH_ANGSTROM)
    assert captured["compute"]["line_wavelength_range_angstrom"] == (10.0, 1000.0)
    assert captured["rasterize"]["emission_measure_convention"] == "ne2"
