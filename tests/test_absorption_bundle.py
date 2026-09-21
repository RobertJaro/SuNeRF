import numpy as np
import pytest
import torch

from sunerf.absorption import ABSORPTION_SPECIES, AbsorptionBundle, load_absorption_bundle
from sunerf.absorption.builder import (
    build_absorption_bundle,
    save_ionization_input,
)
from sunerf.absorption.torch import PhotoionizationOpacity
from sunerf.response.builder import InstrumentThroughput


def _throughput_provenance():
    return {
        "instrument": {"name": "TEST/EUV"},
        "provider": {"name": "test", "version": "1"},
        "calibration": {
            "name": "test-effective-area",
            "version": "1",
            "sha256": "a" * 64,
        },
        "sensitivity_convention": "reference_epoch",
        "radiometry": {
            "measurement_semantics": "surface_brightness",
        },
    }


def _write_inputs(tmp_path):
    verner_path = tmp_path / "photo.dat"
    # Z, N, Eth, Emax, E0, sigma0, ya, P, yw, y0, y1. Values are the
    # H I, He I, and hydrogenic He II rows from the Verner table.
    np.savetxt(
        verner_path,
        np.asarray([
            [1, 1, 13.60, 5.0e4, 0.4298, 5.475e4, 32.88, 2.963, 0, 0, 0],
            [2, 2, 24.59, 5.0e4, 13.61, 949.2, 1.469, 3.188, 2.039, 0.4434, 2.136],
            [2, 1, 54.42, 5.0e4, 1.720, 1.369e4, 32.88, 2.963, 0, 0, 0],
        ]),
    )
    ionization_path = tmp_path / "ionization.npz"
    save_ionization_input(
        ionization_path,
        log_temperature=[4.0, 5.0, 6.0],
        h_i=[0.8, 0.2, 0.0],
        h_ii=[0.2, 0.8, 1.0],
        he_i=[0.7, 0.2, 0.0],
        he_ii=[0.2, 0.6, 0.1],
        he_iii=[0.1, 0.2, 0.9],
        metal_electron_per_hydrogen=[0.01, 0.01, 0.01],
        provenance={"provider": "test", "version": "1"},
    )
    throughput_path = tmp_path / "throughput.npz"
    InstrumentThroughput(
        channels=("171", "304"),
        wavelength_angstrom=np.asarray([100.0, 171.0, 227.0, 304.0, 400.0]),
        throughput=np.asarray([
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]),
        throughput_unit="cm2",
        calibration_epoch="2025-01-01T00:00:00Z",
        provenance=_throughput_provenance(),
    ).save(throughput_path)
    return verner_path, ionization_path, throughput_path


def test_offline_builder_round_trip_and_channel_thresholds(tmp_path):
    verner, ionization, throughput = _write_inputs(tmp_path)
    output = tmp_path / "absorption.npz"
    built = build_absorption_bundle(
        verner_table_path=verner,
        ionization_path=ionization,
        throughputs={"aia": throughput},
        helium_abundance=0.085,
        abundance_provenance={
            "name": "test-coronal",
            "version": "1",
            "sha256": "b" * 64,
        },
        output_path=output,
    )
    loaded = load_absorption_bundle(output)

    assert loaded.bundle_id == built.bundle_id
    assert loaded.species == ABSORPTION_SPECIES
    assert loaded.instrument_keys == ("aia", "aia")
    assert loaded.channels == ("171", "304")
    assert np.all(loaded.effective_cross_section_cm2 >= 0)
    # 304 A photons are below the He II ionization threshold at 228 A.
    assert loaded.effective_cross_section_cm2[1, 2] == pytest.approx(0.0)
    assert loaded.effective_cross_section_cm2[0, 2] > 0


def test_training_provider_is_deterministic_and_channel_specific(tmp_path):
    bundle = AbsorptionBundle(
        species=ABSORPTION_SPECIES,
        log_temperature=np.asarray([4.0, 6.0]),
        ion_fraction=np.asarray([[0.5, 0.5], [0.25, 0.25], [0.125, 0.125]]),
        electron_per_hydrogen=np.asarray([1.0, 1.0]),
        abundance_per_hydrogen=np.asarray([1.0, 0.1, 0.1]),
        instrument_keys=("aia", "aia"),
        channels=("171", "193"),
        effective_cross_section_cm2=np.asarray([
            [2.0e-18, 4.0e-18, 8.0e-18],
            [1.0e-18, 2.0e-18, 4.0e-18],
        ]),
        provenance={"provider": "unit-test"},
    )
    path = tmp_path / "bundle.npz"
    bundle.save(path)
    provider = PhotoionizationOpacity(path, "aia", ["193", "171"])
    total_ne = torch.tensor([[[1.0e8]]])
    result = provider.opacity(
        total_ne=total_ne,
        total_log_ne=torch.log10(total_ne),
        mean_log_T=torch.tensor([[[5.0]]]),
    )

    species_density = torch.tensor([5.0e7, 2.5e6, 1.25e6])
    expected_171 = (species_density * torch.tensor([2e-18, 4e-18, 8e-18])).sum()
    expected_193 = (species_density * torch.tensor([1e-18, 2e-18, 4e-18])).sum()
    torch.testing.assert_close(
        result["alpha_cm_inverse"],
        torch.tensor([[[expected_193, expected_171]]]),
    )
    assert not any(parameter.requires_grad for parameter in provider.parameters())


def test_bundle_rejects_missing_instrument_channel(tmp_path):
    bundle = AbsorptionBundle(
        species=ABSORPTION_SPECIES,
        log_temperature=np.asarray([4.0, 6.0]),
        ion_fraction=np.ones((3, 2)) * 0.1,
        electron_per_hydrogen=np.ones(2),
        abundance_per_hydrogen=np.asarray([1.0, 0.1, 0.1]),
        instrument_keys=("aia",),
        channels=("171",),
        effective_cross_section_cm2=np.ones((1, 3)) * 1e-18,
        provenance={"provider": "unit-test"},
    )
    path = tmp_path / "bundle.npz"
    bundle.save(path)
    with pytest.raises(KeyError, match="no row"):
        PhotoionizationOpacity(path, "euvi_a", ["171"])


def _two_temperature_bundle(tmp_path):
    bundle = AbsorptionBundle(
        species=ABSORPTION_SPECIES,
        log_temperature=np.asarray([4.0, 6.0]),
        ion_fraction=np.asarray([[1.0, 1.0e-6], [1.0, 1.0e-8], [1.0e-8, 1.0e-4]]),
        electron_per_hydrogen=np.asarray([2.0e-3, 1.2]),
        abundance_per_hydrogen=np.asarray([1.0, 0.1, 0.1]),
        instrument_keys=("euvi_a",),
        channels=("171",),
        effective_cross_section_cm2=np.asarray([[1.0e-19, 1.0e-19, 1.0e-19]]),
        provenance={"provider": "unit-test"},
    )
    path = tmp_path / "bundle.npz"
    bundle.save(path)
    return path


def test_hydrogen_density_conventions_and_log_space_ion_fractions(tmp_path):
    path = _two_temperature_bundle(tmp_path)
    state = {
        "total_ne": torch.tensor([[[1.2e9]]]),
        "total_log_ne": torch.log10(torch.tensor([[[1.2e9]]])),
        "mean_log_T": torch.tensor([[[4.0]]]),
    }
    # A configuration key such as ``EUVI-A`` selects the packaged ``euvi_a`` rows.
    proxy = PhotoionizationOpacity(path, "EUVI-A", ["171"])
    cie = PhotoionizationOpacity(
        path, "EUVI-A", ["171"],
        hydrogen_density_convention="cie_electrons_per_hydrogen",
        minimum_electron_per_hydrogen=0.1,
    )

    proxy_state = proxy.opacity(**state)
    cie_state = cie.opacity(**state)

    # Fully ionized proxy: n_H = n_e / 1.2, independent of the local ionization.
    torch.testing.assert_close(
        proxy_state["total_hydrogen_density_cm3"], torch.tensor([[[1.0e9]]])
    )
    # The equilibrium inversion is bounded by its documented floor (not 2e-3).
    torch.testing.assert_close(
        cie_state["total_hydrogen_density_cm3"], torch.tensor([[[1.2e10]]])
    )

    # Midway in log T the H I fraction is the geometric, not arithmetic, mean.
    middle = proxy.opacity(**{**state, "mean_log_T": torch.tensor([[[5.0]]])})
    torch.testing.assert_close(
        middle["absorber_ion_fraction"][0, 0, 0], torch.tensor(1.0e-3), rtol=1e-4, atol=0
    )
    with pytest.raises(ValueError, match="hydrogen_density_convention"):
        PhotoionizationOpacity(path, "euvi_a", ["171"], hydrogen_density_convention="x")


def test_cross_sections_can_be_weighted_by_the_detected_spectrum(tmp_path):
    from sunerf.response.builder import SpectralEmissivityGrid

    verner, ionization, throughput = _write_inputs(tmp_path)
    wavelength = np.asarray([100.0, 150.0, 171.0, 200.0, 304.0, 400.0])
    emissivity = np.zeros((2, wavelength.size))
    emissivity[:, 2] = 1.0  # one line at the 171 A throughput peak
    emissivity[:, 4] = 1.0
    sha = "c" * 64
    spectral_path = tmp_path / "spectral.npz"
    SpectralEmissivityGrid(
        wavelength_angstrom=wavelength,
        log_temperature=np.asarray([5.0, 6.0]),
        emissivity=emissivity,
        emissivity_unit="ph cm3 s-1 sr-1 Angstrom-1",
        emission_measure_convention="ne2",
        provenance={
            "provider": {"name": "test", "version": "1"},
            "atomic_database": {"name": "t", "version": "1", "sha256": sha},
            "abundance": {"name": "t", "version": "1", "sha256": sha},
            "ionization_equilibrium": {"name": "t", "version": "1", "sha256": sha},
            "emission_components": ["lines"],
        },
    ).save(spectral_path)
    common = {
        "verner_table_path": verner, "ionization_path": ionization,
        "throughputs": {"aia": throughput}, "helium_abundance": 0.085,
        "abundance_provenance": {"name": "t", "version": "1", "sha256": "b" * 64},
    }

    plain = build_absorption_bundle(**common)
    weighted = build_absorption_bundle(**common, spectral_emissivity_path=spectral_path)

    from sunerf.absorption.builder import load_verner_parameters, verner_cross_section_cm2

    expected = verner_cross_section_cm2(171.0, load_verner_parameters(verner)["H_I"])
    # A single line inside the passband selects the cross section at that line.
    assert weighted.effective_cross_section_cm2[0, 0] == pytest.approx(float(expected), rel=1e-6)
    assert weighted.provenance["folding"]["method"] == "detected_spectrum_weighted_cross_section"
    assert plain.provenance["folding"]["method"] == "throughput_weighted_cross_section"
    assert weighted.bundle_id != plain.bundle_id


def test_cool_absorber_adds_fixed_wavelength_dependent_opacity(tmp_path):
    bundle = AbsorptionBundle(
        species=ABSORPTION_SPECIES,
        log_temperature=np.asarray([4.0, 7.0]),
        ion_fraction=np.full((3, 2), 1.0e-12),  # the hot plasma itself is transparent
        electron_per_hydrogen=np.asarray([1.2, 1.2]),
        abundance_per_hydrogen=np.asarray([1.0, 0.1, 0.1]),
        instrument_keys=("aia", "aia"),
        channels=("94", "211"),
        effective_cross_section_cm2=np.asarray([
            [1.0e-20, 2.0e-19, 1.0e-19],
            [1.0e-19, 1.4e-18, 1.2e-18],
        ]),
        provenance={"provider": "unit-test"},
    )
    path = tmp_path / "bundle.npz"
    bundle.save(path)
    state = {
        "total_ne": torch.full((1, 2, 1), 1.0e9),
        "total_log_ne": torch.full((1, 2, 1), 9.0),
        "mean_log_T": torch.full((1, 2, 1), 6.2),
    }
    provider = PhotoionizationOpacity(
        path, "aia", ["94", "211"], cool_ion_fractions={"H_I": 0.5, "He_I": 0.8, "He_II": 0.2}
    )

    without = provider.opacity(**state)
    cool = torch.tensor([[[0.0], [1.0e10]]])
    with_cool = provider.opacity(**state, cool_hydrogen_density=cool)

    # kappa_c = sum_s A_s x_s sigma[c, s]; one density sets every channel.
    kappa = torch.tensor([
        0.5 * 1.0e-20 + 0.1 * (0.8 * 2.0e-19 + 0.2 * 1.0e-19),
        0.5 * 1.0e-19 + 0.1 * (0.8 * 1.4e-18 + 0.2 * 1.2e-18),
    ])
    torch.testing.assert_close(provider.cool_cross_section_per_hydrogen_cm2, kappa)
    torch.testing.assert_close(
        with_cool["alpha_cm_inverse"][0, 1] - without["alpha_cm_inverse"][0, 1], 1.0e10 * kappa
    )
    torch.testing.assert_close(
        with_cool["alpha_cm_inverse"][0, 0], without["alpha_cm_inverse"][0, 0]
    )
    assert "cool_hydrogen_density_cm3" not in without
    # The channel ratio is fixed by atomic physics, not learned.
    assert (kappa[1] / kappa[0]).item() == pytest.approx(0.186 / 0.023, rel=1e-3)

    assert PhotoionizationOpacity(path, "aia", ["94"]).cool_ion_fractions == {
        "H_I": 0.7, "He_I": 0.7, "He_II": 0.3,
    }
    with pytest.raises(ValueError, match="must not exceed one"):
        PhotoionizationOpacity(
            path, "aia", ["94"], cool_ion_fractions={"H_I": 0.5, "He_I": 0.8, "He_II": 0.5}
        )
    with pytest.raises(ValueError, match="define exactly"):
        PhotoionizationOpacity(path, "aia", ["94"], cool_ion_fractions={"H_I": 0.5})
