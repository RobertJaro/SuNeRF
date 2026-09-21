import numpy as np
import pytest
import torch

from sunerf.rendering.plasma import PlasmaRadiativeTransfer
from sunerf.response import load_response_artifact
from sunerf.response.builder import (
    InstrumentThroughput,
    SpectralEmissivityGrid,
    build_response_artifact,
    fold_temperature_response,
    input_schema_description,
    load_instrument_throughput,
    load_spectral_emissivity,
)


def _spectral_provenance():
    return {
        "provider": {"name": "fiasco", "version": "0.test"},
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
            "name": "chianti.ioneq",
            "version": "11.0.2",
            "sha256": "c" * 64,
        },
        "emission_components": ["lines", "free_free", "free_bound", "two_photon"],
        "hydrogen_to_electron_ratio": 0.82,
    }


def _throughput_provenance():
    return {
        "instrument": {"name": "TEST/EUV"},
        "provider": {"name": "test-calibration", "version": "2.1"},
        "calibration": {
            "name": "effective-area-table",
            "version": "2025-01",
            "sha256": "d" * 64,
        },
        "sensitivity_convention": "reference_epoch",
        "radiometry": {
            "measurement_semantics": "per_native_pixel",
            "native_pixel_solid_angle_sr": 8.461594994075237e-12,
            "native_pixel_solid_angle_relative_tolerance": 0.05,
        },
    }


def _spectral(*, density_dependent=False):
    emissivity = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    log_density = None
    if density_dependent:
        log_density = np.array([8.0, 10.0])
        emissivity = np.stack([emissivity, 2.0 * emissivity])
    return SpectralEmissivityGrid(
        wavelength_angstrom=np.array([10.0, 11.0, 13.0]),
        log_temperature=np.array([5.0, 6.0]),
        log_density=log_density,
        emissivity=emissivity,
        emissivity_unit="ph cm3 s-1 sr-1 Angstrom-1",
        emission_measure_convention="ne_nh",
        provenance=_spectral_provenance(),
    )


def _throughput():
    return InstrumentThroughput(
        channels=("A", "B"),
        wavelength_angstrom=np.array([10.0, 11.0, 13.0]),
        throughput=np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]]),
        throughput_unit="cm2 sr DN ph-1 pix-1",
        calibration_epoch="2025-01-02T03:04:05+00:00",
        provenance=_throughput_provenance(),
    )


def test_fold_shared_spectrum_preserves_channel_order_units_and_provenance():
    spectral = _spectral()
    throughput = _throughput()

    response = fold_temperature_response(
        spectral, throughput, channels=("B", "A")
    )
    response_again = fold_temperature_response(
        spectral, throughput, channels=("B", "A")
    )

    assert response.channels == ("B", "A")
    assert response.emission_measure_convention == "ne_nh"
    # The nonuniform wavelength weights integrate the [10, 13] interval to 3 A.
    np.testing.assert_allclose(response.response, [[6.0, 12.0], [18.0, 36.0]])
    assert response.response_unit == "cm5 DN / (pix s)"
    assert response.provenance == response_again.provenance
    assert len(response.provenance["spectral_emissivity"]["content_sha256"]) == 64
    assert response.provenance["spectral_emissivity"]["abundance"]["name"] == (
        "sun_coronal_2021_chianti"
    )
    assert response.provenance["instrument_throughput"]["calibration_epoch"].endswith("Z")
    assert response.provenance["fold"]["normalization"] == "none"
    assert response.provenance["hydrogen_to_electron_ratio"] == 0.82
    assert response.provenance["sensitivity_convention"] == "reference_epoch"
    assert response.provenance["calibration_epoch"] == "2025-01-02T03:04:05Z"
    assert response.provenance["measurement_semantics"] == "per_native_pixel"
    assert response.provenance["native_pixel_solid_angle_sr"] > 0
    assert response.response_id.startswith("sha256:")
    assert response.verify_response_id()


def test_density_axis_is_preserved_by_fold():
    response = fold_temperature_response(_spectral(density_dependent=True), _throughput())

    assert response.response.shape == (2, 2, 2)
    np.testing.assert_allclose(response.log_density, [8.0, 10.0])
    np.testing.assert_allclose(response.response[0, 1], [36.0, 72.0])


def test_validated_input_roundtrip_and_api_write_response_npz(tmp_path):
    spectral_path = tmp_path / "chianti-emissivity.npz"
    throughput_path = tmp_path / "instrument-throughput.npz"
    response_path = tmp_path / "response.npz"
    _spectral().save(spectral_path)
    _throughput().save(throughput_path)

    assert load_spectral_emissivity(spectral_path).content_sha256 == _spectral().content_sha256
    assert load_instrument_throughput(throughput_path).channels == ("A", "B")
    build_response_artifact(
        spectral_path,
        throughput_path,
        response_path,
        channels=("B",),
    )

    loaded = load_response_artifact(response_path)
    assert loaded.channels == ("B",)
    np.testing.assert_allclose(loaded.response, [[6.0, 12.0]])

    api_path = tmp_path / "api-response.npz"
    built = build_response_artifact(spectral_path, throughput_path, api_path)
    assert api_path.is_file()
    assert built.channels == ("A", "B")


def test_builder_artifact_loads_into_physical_renderer_end_to_end(tmp_path):
    spectral_path = tmp_path / "chianti-emissivity.npz"
    throughput_path = tmp_path / "instrument-throughput.npz"
    response_path = tmp_path / "response.npz"
    _spectral().save(spectral_path)
    _throughput().save(throughput_path)

    source = build_response_artifact(
        spectral_path,
        throughput_path,
        response_path,
        channels=("A",),
    )
    renderer = PlasmaRadiativeTransfer(
        {
            "artifact": str(response_path),
            "channels": ["A"],
            "model_length_unit_cm": 1.0,
        },
        np.array([5.0, 6.0], dtype=np.float32),
    ).eval()

    log_ne = torch.full((1, 2, 2), -30.0)
    log_ne[..., 1] = np.log10(2.0)
    output = renderer(
        log_ne=log_ne,
        total_ne=torch.full((1, 2, 1), 2.0),
        mean_log_T=torch.full((1, 2, 1), 6.0),
        total_log_ne=torch.full((1, 2, 1), np.log10(2.0)),
        z_vals=torch.tensor([[0.0, 1.0]]),
        rays_d=torch.tensor([[1.0, 0.0, 0.0]]),
        query_points=torch.zeros((1, 2, 4)),
    )

    # K_A(logT=6)=36 cm5 DN/(pix s), n_e^2=4 cm^-6,
    # n_H/n_e=0.82, and the physical path is exactly 1 cm.
    torch.testing.assert_close(output["image"], torch.tensor([[118.08]]))
    assert renderer.channels == ("A",)
    assert renderer.response_id == source.response_id
    assert renderer.response_provenance["hydrogen_to_electron_ratio"] == 0.82


def test_builder_rejects_incomplete_physics_metadata_and_ambiguous_epoch():
    provenance = _spectral_provenance()
    provenance.pop("abundance")
    with pytest.raises(ValueError, match="provenance.abundance"):
        SpectralEmissivityGrid(
            wavelength_angstrom=[10.0, 11.0],
            log_temperature=[5.0, 6.0],
            emissivity=np.ones((2, 2)),
            emissivity_unit="ph cm3 s-1 Angstrom-1",
            emission_measure_convention="ne2",
            provenance=provenance,
        )

    with pytest.raises(ValueError, match="UTC offset"):
        InstrumentThroughput(
            channels=("A",),
            wavelength_angstrom=[10.0, 11.0],
            throughput=np.ones((1, 2)),
            throughput_unit="cm2",
            calibration_epoch="2025-01-02T03:04:05",
            provenance=_throughput_provenance(),
        )

    throughput_provenance = _throughput_provenance()
    throughput_provenance["sensitivity_convention"] = "unspecified"
    with pytest.raises(ValueError, match="sensitivity_convention"):
        InstrumentThroughput(
            channels=("A",),
            wavelength_angstrom=[10.0, 11.0],
            throughput=np.ones((1, 2)),
            throughput_unit="cm2",
            calibration_epoch="2025-01-02T03:04:05Z",
            provenance=throughput_provenance,
        )


def test_builder_rejects_nonoverlap_and_incompatible_requested_unit():
    throughput = InstrumentThroughput(
        channels=("A",),
        wavelength_angstrom=[20.0, 21.0],
        throughput=np.ones((1, 2)),
        throughput_unit="cm2 pix-1",
        calibration_epoch="2025-01-02T03:04:05Z",
        provenance=_throughput_provenance(),
    )
    with pytest.raises(ValueError, match="do not overlap"):
        fold_temperature_response(_spectral(), throughput)
    with pytest.raises(ValueError, match="incompatible"):
        fold_temperature_response(_spectral(), _throughput(), response_unit="kg")


def test_throughput_requires_radiometry_and_pixel_unit_consistency():
    provenance = _throughput_provenance()
    provenance.pop("radiometry")
    with pytest.raises(ValueError, match="radiometry"):
        InstrumentThroughput(
            channels=("A",),
            wavelength_angstrom=[10.0, 11.0],
            throughput=np.ones((1, 2)),
            throughput_unit="cm2 pix-1",
            calibration_epoch="2025-01-02T03:04:05Z",
            provenance=provenance,
        )

    provenance = _throughput_provenance()
    provenance["radiometry"] = {"measurement_semantics": "surface_brightness"}
    with pytest.raises(ValueError, match="must not contain a pixel unit"):
        InstrumentThroughput(
            channels=("A",),
            wavelength_angstrom=[10.0, 11.0],
            throughput=np.ones((1, 2)),
            throughput_unit="cm2 pix-1",
            calibration_epoch="2025-01-02T03:04:05Z",
            provenance=provenance,
        )


def test_schema_describes_provider_boundary():
    schema = input_schema_description()

    assert schema["spectral_emissivity_npz"]["schema"] == (
        "sunerf.chianti-spectral-emissivity"
    )
    conventions = schema["instrument_throughput_npz"]["required_provenance"][
        "sensitivity_convention"
    ]
    assert conventions == ["native_epoch", "reference_epoch", "static_assumed"]
