import unittest

import numpy as np
import torch
from astropy import units as u

from sunerf.data.loader.volume_sampling import RandomSphericalCoordinateDataset


class RandomSphericalCoordinateDatasetTest(unittest.TestCase):

    @staticmethod
    def _dataset(**kwargs):
        config = {
            "radius_range": np.array([1.0, 3.0]) * u.Rsun,
            "latitude_range": np.array([-30.0, 60.0]) * u.deg,
            "longitude_range": np.array([20.0, 140.0]) * u.deg,
            "time_range": [-2.0, 2.0],
            "batch_size": 5,
        }
        config.update(kwargs)
        return RandomSphericalCoordinateDataset(**config)

    def test_volume_uniform_sampling_is_the_default(self):
        self.assertTrue(self._dataset().volume_uniform_sampling)

    def test_samples_are_uniform_in_physical_volume_coordinates(self):
        unit_samples = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75, 0.75],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float32)
        dataset = self._dataset()
        dataset._sample_unit_coordinates = lambda: unit_samples

        samples = dataset[0]["coords"]
        xyz = samples[:, :3]
        radius = torch.linalg.vector_norm(xyz, dim=1)
        latitude = torch.asin(xyz[:, 2] / radius)
        longitude = torch.atan2(xyz[:, 1], xyz[:, 0])

        normalized_volume = (radius**3 - 1.0**3) / (3.0**3 - 1.0**3)
        sin_lat_min = np.sin(np.deg2rad(-30.0))
        sin_lat_max = np.sin(np.deg2rad(60.0))
        normalized_solid_angle = (torch.sin(latitude) - sin_lat_min) / (sin_lat_max - sin_lat_min)
        normalized_longitude = (longitude - np.deg2rad(20.0)) / np.deg2rad(120.0)

        torch.testing.assert_close(normalized_volume, unit_samples[:, 0], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(normalized_solid_angle, unit_samples[:, 1], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(normalized_longitude, unit_samples[:, 2], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(samples[:, 3], -2.0 + 4.0 * unit_samples[:, 3])

    def test_legacy_coordinate_uniform_sampling_remains_available(self):
        dataset = self._dataset(volume_uniform_sampling=False)
        self.assertFalse(dataset.volume_uniform_sampling)

    def test_radial_sampling_exponent_biases_toward_lower_radius(self):
        unit_samples = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75, 0.75],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float32)
        dataset = self._dataset(radial_sampling_exponent=2)
        dataset._sample_unit_coordinates = lambda: unit_samples

        samples = dataset[0]["coords"]
        radius = torch.linalg.vector_norm(samples[:, :3], dim=1)
        normalized_radius = (radius - 1.0) / (3.0 - 1.0)

        torch.testing.assert_close(normalized_radius, unit_samples[:, 0].pow(2), atol=2e-6, rtol=2e-6)

    def test_radial_sampling_exponent_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self._dataset(radial_sampling_exponent=0)

    def test_radial_strata_guarantee_exact_shell_counts(self):
        dataset = self._dataset(
            batch_size=7,
            radial_strata={
                "edges": [1.0, 1.5, 2.0, 3.0],
                "fractions": [0.25, 0.25, 0.5],
            },
        )
        dataset._sample_unit_coordinates = lambda: torch.full((7, 4), 0.5)

        radius = torch.linalg.vector_norm(dataset[0]["coords"][:, :3], dim=1)

        # Largest-remainder allocation is stable: [1.75, 1.75, 3.5]
        # becomes exactly [2, 2, 3].
        self.assertEqual([2, 2, 3], dataset._stratum_counts().tolist())
        self.assertEqual(2, int((radius < 1.5).sum()))
        self.assertEqual(2, int(((radius >= 1.5) & (radius < 2.0)).sum()))
        self.assertEqual(3, int((radius >= 2.0).sum()))

    def test_radial_strata_are_volume_uniform_within_each_shell(self):
        dataset = self._dataset(
            batch_size=4,
            radial_strata={
                "edges": [1.0, 2.0, 3.0],
                "fractions": [0.5, 0.5],
            },
        )
        unit_samples = torch.tensor([
            [0.0, 0.5, 0.5, 0.5],
            [1.0, 0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5, 0.5],
            [1.0, 0.5, 0.5, 0.5],
        ])
        dataset._sample_unit_coordinates = lambda: unit_samples

        radius = torch.linalg.vector_norm(dataset[0]["coords"][:, :3], dim=1)
        torch.testing.assert_close(radius, torch.tensor([1.0, 2.0, 2.0, 3.0]))

    def test_radial_strata_validate_coverage_and_conflicts(self):
        with self.assertRaisesRegex(ValueError, "span"):
            self._dataset(radial_strata={"edges": [1.2, 2.0, 3.0], "fractions": [0.5, 0.5]})
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self._dataset(
                radial_sampling_exponent=2,
                radial_strata={"edges": [1.0, 2.0, 3.0], "fractions": [0.5, 0.5]},
            )


if __name__ == "__main__":
    unittest.main()
