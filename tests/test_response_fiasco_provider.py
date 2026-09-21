import hashlib

import h5py
import numpy as np
import pytest

from sunerf.response.builder import InstrumentThroughput, fold_temperature_response
from sunerf.response.numerics import trapezoid_node_weights
from sunerf.response.providers.fiasco import (
    ContinuumEmissivity,
    ExactLineEmissivity,
    FiascoConfig,
    FiascoEmissionModel,
    inspect_fiasco_sources,
    rasterize_lines,
)


def _spectral_provenance():
    return {
        "provider": {"name": "fiasco", "version": "0.8.2"},
        "atomic_database": {
            "name": "CHIANTI",
            "version": "11.0.2",
            "sha256": "a" * 64,
        },
        "abundance": {
            "name": "sun_coronal_2021_chianti",
            "version": "11.0.2",
            "sha256": "b" * 64,
        },
        "ionization_equilibrium": {
            "name": "chianti",
            "version": "11.0.2",
            "sha256": "c" * 64,
        },
        "emission_components": ["lines", "free_free", "free_bound", "two_photon"],
    }


def _throughput_provenance():
    return {
        "instrument": {"name": "TEST/EUV"},
        "provider": {"name": "test", "version": "1"},
        "calibration": {
            "name": "test-effective-area",
            "version": "1",
            "sha256": "d" * 64,
        },
        "sensitivity_convention": "reference_epoch",
        "radiometry": {
            "measurement_semantics": "per_native_pixel",
            "native_pixel_solid_angle_sr": 1e-11,
            "native_pixel_solid_angle_relative_tolerance": 0.02,
        },
    }


def _model():
    line = np.array([[[2.0, 1.0], [4.0, 3.0]]])
    continuum = np.ones((1, 2, 3))
    zeros = np.zeros_like(continuum)
    return FiascoEmissionModel(
        log_temperature=np.array([5.0, 6.0]),
        log_density=np.array([9.0]),
        hydrogen_to_electron_ratio=np.array([0.8, 0.5]),
        lines=ExactLineEmissivity(
            wavelength_angstrom=np.array([10.5, 12.0]),
            ion=("Fe 9", "Fe 12"),
            emissivity=line,
        ),
        continuum=ContinuumEmissivity(
            wavelength_angstrom=np.array([10.0, 11.0, 13.0]),
            free_free=continuum,
            free_bound=zeros,
            two_photon=zeros,
        ),
        provenance=_spectral_provenance(),
    )


def _throughput():
    return InstrumentThroughput(
        channels=("A", "B"),
        wavelength_angstrom=np.array([10.0, 11.0, 13.0]),
        throughput=np.array([[1.0, 2.0, 4.0], [2.0, 2.0, 2.0]]),
        throughput_unit="cm2 sr DN / (ph pix)",
        calibration_epoch="2020-01-01T00:00:00Z",
        provenance=_throughput_provenance(),
    )


def test_rasterize_lines_conserves_integrated_photons_and_linear_fold():
    wavelength = np.array([10.0, 11.0, 13.0])
    line_wavelength = np.array([10.0, 10.5, 12.5, 13.0])
    integrated = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    raster = rasterize_lines(line_wavelength, integrated, wavelength)
    weights = trapezoid_node_weights(wavelength)

    np.testing.assert_allclose(np.einsum("rw,w->r", raster, weights), integrated.sum(axis=1))

    linear_response = np.array([1.0, 3.0, 7.0])
    raster_fold = np.einsum("rw,w,w->r", raster, linear_response, weights)
    exact_fold = np.einsum(
        "rl,l->r",
        integrated,
        np.interp(line_wavelength, wavelength, linear_response),
    )
    np.testing.assert_allclose(raster_fold, exact_fold)


def test_exact_hybrid_fold_uses_temperature_dependent_ne2_conversion():
    response = _model().fold(_throughput())

    assert response.channels == ("A", "B")
    assert response.emission_measure_convention == "ne2"
    assert response.response_unit == "cm5 DN / (pix s)"
    np.testing.assert_allclose(
        response.response[:, 0, :],
        np.array([[10.8, 11.25], [9.6, 10.0]]),
    )
    assert response.provenance["fold"]["line_normalization"] == (
        "integrated_photon_emissivity_no_kernel"
    )
    assert response.provenance["spectral_emissivity"]["emission_measure_export"][
        "ne2_conversion"
    ] == "temperature-dependent fiasco.proton_electron_ratio"


def test_provider_ne_nh_export_is_explicit_and_unscaled():
    with pytest.raises(ValueError, match="explicit scalar"):
        _model().fold(_throughput(), emission_measure_convention="ne_nh")

    response = _model().fold(
        _throughput(),
        emission_measure_convention="ne_nh",
        hydrogen_to_electron_ratio=0.82,
    )
    np.testing.assert_allclose(
        response.response[:, 0, :],
        np.array([[13.5, 22.5], [12.0, 20.0]]),
    )
    assert response.provenance["hydrogen_to_electron_ratio"] == 0.82


def test_rasterized_provider_grid_matches_exact_fold_on_same_nodes():
    model = _model()
    throughput = _throughput()
    exact = model.fold(throughput)
    spectral = model.rasterize(wavelength_angstrom=throughput.wavelength_angstrom)
    rasterized = fold_temperature_response(spectral, throughput)

    np.testing.assert_allclose(rasterized.response, exact.response)
    assert spectral.emission_measure_convention == "ne2"
    assert spectral.provenance["spectral_export"]["excluded_line_count"] == 0


def test_source_inspection_pins_ascii_inputs_and_hdf5_selected_datasets(tmp_path):
    database_path = tmp_path / "chianti-11.0.2.h5"
    abundance_path = tmp_path / "sun_coronal_2021_chianti.abund"
    ionization_path = tmp_path / "chianti.ioneq"
    abundance_path.write_text("abundance source\n", encoding="ascii")
    ionization_path.write_text("ionization source\n", encoding="ascii")
    with h5py.File(database_path, "w") as database:
        abundance = database.create_group("h/abundance")
        abundance.attrs["chianti_version"] = "11.0.2"
        abundance.create_dataset("sun_coronal_2021_chianti", data=1.0)
        ionization = database.create_group("h/h_1/ioneq/chianti")
        ionization.attrs["chianti_version"] = "11.0.2"
        ionization.create_dataset("ionization_fraction", data=[1.0])
        ionization.create_dataset("temperature", data=[1e6])
        potential = database.create_group("h/h_1/ip")
        potential.attrs["chianti_version"] = "11.0.2"
        potential.create_dataset("chianti", data=1.0)

    config = FiascoConfig(
        hdf5_database=database_path,
        abundance_file=abundance_path,
        ionization_equilibrium_file=ionization_path,
    )
    provenance = inspect_fiasco_sources(config)

    assert provenance["provider"] == {"name": "fiasco", "version": "0.8.2"}
    assert provenance["atomic_database"]["version"] == "11.0.2"
    assert provenance["atomic_database"]["sha256"] == hashlib.sha256(
        database_path.read_bytes()
    ).hexdigest()
    assert provenance["abundance"]["sha256"] == hashlib.sha256(
        abundance_path.read_bytes()
    ).hexdigest()
    assert provenance["ionization_equilibrium"]["sha256"] == hashlib.sha256(
        ionization_path.read_bytes()
    ).hexdigest()

    bad_config = FiascoConfig(
        hdf5_database=database_path,
        abundance_file=abundance_path,
        ionization_equilibrium_file=ionization_path,
        expected_database_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        inspect_fiasco_sources(bad_config)


def test_config_rejects_dataset_name_source_file_mismatch(tmp_path):
    with pytest.raises(ValueError, match="abundance_file stem"):
        FiascoConfig(
            hdf5_database=tmp_path / "db.h5",
            abundance_file=tmp_path / "wrong.abund",
            ionization_equilibrium_file=tmp_path / "chianti.ioneq",
        )
