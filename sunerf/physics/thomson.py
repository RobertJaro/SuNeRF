import numpy as np

MSB = 4.67E+20  # ph / cm^2 / s / sr
SIGMA_NE = 7.95e-26
R_SUN_CM = 6.957e10
LIMB_DARKENING_COEFF = 0.63


def electron_density_normalization_cm3(msb_norm, Rs_per_ds, sigma_ne=SIGMA_NE):
    r"""Convert dimensionless model density to ``cm^-3`` for MSB data.

    The renderer implements the van de Hulst kernels without their dimensional
    prefactor.  For brightness divided by the *mean* solar-disk brightness, the
    prefactor is

    ``(pi * sigma_ne / 2) / (1 - u / 3)``.

    Consequently the density scale contains ``1 - u / 3`` in its numerator.
    ``u`` is kept equal to the renderer's white-light limb-darkening coefficient.
    """
    ds_cm = float(Rs_per_ds) * R_SUN_CM
    thomson_factor = np.pi * float(sigma_ne) / 2.0
    mean_disk_factor = 1.0 - LIMB_DARKENING_COEFF / 3.0
    return float(msb_norm) * mean_disk_factor / (thomson_factor * ds_cm)
