"""Training-side ingestion of immutable absorption bundles."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sunerf.absorption import load_absorption_bundle


HYDROGEN_DENSITY_CONVENTIONS = frozenset({
    "fully_ionized_proxy", "cie_electrons_per_hydrogen",
})
# Fractions of neutral hydrogen, neutral helium and singly ionized helium in
# cool, photoionized prominence/filament plasma. They are an assumption, not an
# equilibrium result: a hydrogen ionization degree of 0.3 and 30 % He II are
# the values commonly adopted for EUV absorption mass estimates (Anzer &
# Heinzel 2005; Williams et al. 2013). The inferred hydrogen column scales
# roughly inversely with the neutral fractions.
DEFAULT_COOL_ION_FRACTIONS = {"H_I": 0.7, "He_I": 0.7, "He_II": 0.3}


def _validated_cool_ion_fractions(fractions, species):
    fractions = dict(DEFAULT_COOL_ION_FRACTIONS if fractions is None else fractions)
    if set(fractions) != set(species):
        raise ValueError(f"cool_ion_fractions must define exactly {list(species)}")
    values = [float(fractions[name]) for name in species]
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("cool_ion_fractions must lie in [0, 1]")
    if fractions["He_I"] + fractions["He_II"] > 1.0 + 1e-9:
        raise ValueError("cool He I and He II fractions must not exceed one in total")
    return values


class PhotoionizationOpacity(nn.Module):
    """Deterministic H/He opacity selected for one instrument's channels."""

    is_deterministic_physical = True

    def __init__(
        self,
        artifact,
        instrument_key,
        channels,
        hydrogen_density_convention="fully_ionized_proxy",
        minimum_electron_per_hydrogen=0.1,
        cool_ion_fractions=None,
    ):
        super().__init__()
        if hydrogen_density_convention not in HYDROGEN_DENSITY_CONVENTIONS:
            raise ValueError(
                "hydrogen_density_convention must be one of "
                f"{sorted(HYDROGEN_DENSITY_CONVENTIONS)}"
            )
        minimum_electron_per_hydrogen = float(minimum_electron_per_hydrogen)
        if not 0 < minimum_electron_per_hydrogen <= 1:
            raise ValueError("minimum_electron_per_hydrogen must lie in (0, 1]")
        self.hydrogen_density_convention = hydrogen_density_convention
        self.minimum_electron_per_hydrogen = minimum_electron_per_hydrogen
        bundle = load_absorption_bundle(artifact)
        indices = bundle.indices_for(instrument_key, channels)
        self.bundle_id = bundle.bundle_id
        self.instrument_key = str(instrument_key)
        self.channels = tuple(str(channel) for channel in channels)
        self.species = bundle.species
        self.provenance = dict(bundle.provenance)
        self.register_buffer(
            "log_temperature_axis",
            torch.as_tensor(bundle.log_temperature, dtype=torch.float32),
        )
        # Ion fractions span many decades between neighbouring nodes, so they
        # are interpolated in log space.
        self.register_buffer(
            "log_ion_fraction_table",
            torch.log10(
                torch.as_tensor(bundle.ion_fraction, dtype=torch.float64).clamp_min(1.0e-30)
            ).to(torch.float32),
        )
        # Fully ionized limit of the bundled composition (1 + 2 A_He + metals).
        self.register_buffer(
            "fully_ionized_electron_per_hydrogen",
            torch.as_tensor(
                float(np.max(bundle.electron_per_hydrogen)), dtype=torch.float32
            ),
        )
        self.register_buffer(
            "electron_per_hydrogen_table",
            torch.as_tensor(bundle.electron_per_hydrogen, dtype=torch.float32),
        )
        self.register_buffer(
            "abundance_per_hydrogen",
            torch.as_tensor(bundle.abundance_per_hydrogen, dtype=torch.float32),
        )
        self.cool_ion_fractions = dict(zip(
            bundle.species, _validated_cool_ion_fractions(cool_ion_fractions, bundle.species)
        ))
        self.register_buffer(
            "cool_ion_fraction",
            torch.as_tensor(list(self.cool_ion_fractions.values()), dtype=torch.float32),
        )
        self.register_buffer(
            "effective_cross_section_cm2",
            torch.as_tensor(
                bundle.effective_cross_section_cm2[indices], dtype=torch.float32
            ),
        )

    @property
    def cool_cross_section_per_hydrogen_cm2(self):
        """Per-channel cross section of the cool absorber per hydrogen nucleus."""
        return torch.einsum(
            "s,cs->c",
            self.cool_ion_fraction * self.abundance_per_hydrogen,
            self.effective_cross_section_cm2,
        )

    def _interpolate(self, table, log_temperature):
        values = log_temperature.squeeze(-1).clamp(
            self.log_temperature_axis[0], self.log_temperature_axis[-1]
        )
        flat = values.reshape(-1)
        upper = torch.searchsorted(
            self.log_temperature_axis, flat.contiguous(), right=True
        ).clamp(1, self.log_temperature_axis.numel() - 1)
        lower = upper - 1
        lower_temperature = self.log_temperature_axis[lower]
        upper_temperature = self.log_temperature_axis[upper]
        fraction = (flat - lower_temperature) / (
            upper_temperature - lower_temperature
        )
        # Tables use (feature, temperature); indexing the transposed view gives
        # (sample, feature) and retains gradients through the interpolation fraction.
        lower_values = table.transpose(0, 1)[lower]
        upper_values = table.transpose(0, 1)[upper]
        interpolated = torch.lerp(lower_values, upper_values, fraction[:, None])
        return interpolated.reshape(*values.shape, table.shape[0])

    def opacity(
        self,
        *,
        total_ne,
        total_log_ne,
        mean_log_T,
        total_hydrogen_density=None,
        cool_hydrogen_density=None,
    ):
        """Opacity of the emitting plasma plus an optional separate cool absorber.

        ``cool_hydrogen_density`` is the total hydrogen density of cool material
        that does not emit in the EUV channels. Its ionization state is the fixed
        ``cool_ion_fractions``; the cross sections and abundances are the same
        atomic data used for the emitting plasma, so one density field sets the
        opacity of every channel.
        """
        fractions = torch.pow(
            10.0, self._interpolate(self.log_ion_fraction_table, mean_log_T)
        )
        if total_hydrogen_density is None:
            if self.hydrogen_density_convention == "fully_ionized_proxy":
                # The density field traces the total hydrogen density as in a
                # fully ionized single-fluid model; neutral fractions only
                # select how much of that hydrogen absorbs.
                electron_per_hydrogen = self.fully_ionized_electron_per_hydrogen
            else:
                # Inverting n_e through the equilibrium ionization degree is
                # ill-conditioned for nearly neutral plasma; the documented
                # floor bounds the implied hydrogen density.
                electron_per_hydrogen = self._interpolate(
                    self.electron_per_hydrogen_table[None], mean_log_T
                ).clamp_min(self.minimum_electron_per_hydrogen)
            hydrogen_density = total_ne / electron_per_hydrogen
        else:
            hydrogen_density = torch.as_tensor(
                total_hydrogen_density, dtype=total_ne.dtype, device=total_ne.device
            )
            if hydrogen_density.shape != total_ne.shape:
                raise ValueError(
                    "total_hydrogen_density must have the same shape as total_ne"
                )
        hydrogen_density = torch.nan_to_num(
            hydrogen_density, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        species_density = (
            hydrogen_density * fractions * self.abundance_per_hydrogen
        )
        alpha = torch.einsum(
            "...s,cs->...c", species_density, self.effective_cross_section_cm2
        )
        cool_state = {}
        if cool_hydrogen_density is not None:
            cool_density = torch.nan_to_num(
                cool_hydrogen_density, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
            if cool_density.shape != total_ne.shape:
                raise ValueError("cool_hydrogen_density must have the same shape as total_ne")
            cool_alpha = cool_density * self.cool_cross_section_per_hydrogen_cm2
            alpha = alpha + cool_alpha
            cool_state = {
                "cool_hydrogen_density_cm3": cool_density,
                "cool_alpha_cm_inverse": cool_alpha,
            }
        return {
            **cool_state,
            "alpha_cm_inverse": alpha,
            "total_hydrogen_density_cm3": hydrogen_density,
            "absorber_species_density_cm3": species_density,
            "absorber_ion_fraction": fractions,
        }
