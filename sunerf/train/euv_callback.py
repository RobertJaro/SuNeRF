"""Configurable, artifact-driven validation plots for EUV tomography."""

from __future__ import annotations

import math

import numpy as np
import wandb
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity

from sunerf.configuration import canonical_channel_id
from sunerf.train.callback import BaseCallback


_DIAGNOSTIC_FIELDS = {
    "mean_log_temperature": ("mean_T", "Mean log$_{10}$(T / K)", "plasma", "linear"),
    "column_electron_density": (
        "column_electron_density_cm2",
        "Column $N_e$ [cm$^{-2}$]",
        "viridis",
        "log",
    ),
    "emission_measure": (
        "emission_measure_cm5",
        "Emission measure [cm$^{-5}$]",
        "magma",
        "log",
    ),
    "emission_height": (
        "height_map",
        r"Emission-weighted radius [$R_\odot$]",
        "cividis",
        "linear",
    ),
    "absorption_fraction": (
        "mean_absorption",
        "Mean absorbed fraction",
        "cool",
        "fraction",
    ),
    "cool_hydrogen_column": (
        "cool_hydrogen_column_cm2",
        "Cool H column [cm$^{-2}$]",
        "bone",
        "log",
    ),
}


# Mean temperature and emission height are only meaningful where plasma
# emits: pixels whose emission measure is below this fraction of the image's
# 99th percentile are masked in those two panels.
_EMISSION_MEASURE_FLOOR = 1.0e-3
_EMISSION_WEIGHTED_DIAGNOSTICS = ("mean_log_temperature", "emission_height")


def _emission_measure_mask(emission_measure, image_mask):
    """Pixels with enough emission measure for emission-weighted diagnostics."""
    values = _finite_values(emission_measure, image_mask, positive=True)
    if values.size == 0:
        return image_mask
    floor = _EMISSION_MEASURE_FLOOR * np.percentile(values, 99.0)
    with np.errstate(invalid="ignore"):
        return image_mask & (np.nan_to_num(emission_measure, nan=0.0) >= floor)


def _available_diagnostics(quantities, outputs, renderer):
    """Drop diagnostics whose physics is not part of the model."""
    available = []
    for quantity in quantities:
        field = _DIAGNOSTIC_FIELDS[quantity][0]
        if quantity == "absorption_fraction" and not getattr(renderer, "absorption", False):
            continue
        if field not in outputs:
            continue
        available.append(quantity)
    return available


def _append_colorbar(fig, ax, image):
    divider = make_axes_locatable(ax)
    colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(image, cax=colorbar_axis)


def _finite_values(values, mask=None, *, positive=False):
    values = np.asarray(values)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.broadcast_to(mask, values.shape)
    if positive:
        valid &= values > 0
    return values[valid]


def _linear_norm(values, mask=None, percentile=99.5):
    finite = _finite_values(values, mask)
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    low, high = np.nanpercentile(finite, [1.0, percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = float(np.nanmax(finite))
    if high <= low:
        high = low + max(abs(low), 1.0) * 1e-6
    return Normalize(vmin=float(low), vmax=float(high))


def _positive_log_norm(values, mask=None, percentile=99.5):
    finite = _finite_values(values, mask, positive=True)
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    low, high = np.nanpercentile(finite, [1.0, percentile])
    low = max(float(low), np.finfo(np.float32).tiny)
    high = max(float(high), low * (1.0 + 1e-6))
    return LogNorm(vmin=low, vmax=high)


def _match_mean_and_std(values, reference):
    """Affine-rescale ``values`` so its mean and std over finite pixels match
    ``reference``; a constant input is shifted to the reference mean."""
    finite = np.isfinite(values) & np.isfinite(reference)
    if not finite.any():
        return values
    mean = float(np.mean(values[finite]))
    std = float(np.std(values[finite]))
    reference_mean = float(np.mean(reference[finite]))
    reference_std = float(np.std(reference[finite]))
    scale = reference_std / std if std > 0 else 1.0
    return (values - mean) * scale + reference_mean


# Smallest image side for the default 7x7 SSIM window.
_SSIM_MIN_SIDE = 7


def _image_metrics(observed, predicted, mask):
    """MSE, PSNR [dB] and SSIM of one full channel image over ``mask``.

    Images are in loss space (instrument scaling maps them to [0, 1]), so the
    data range is 1 and the PSNR matches ``train/psnr``. SSIM is computed on
    the full image with masked pixels zeroed and averaged over ``mask``.
    """
    mask = mask & np.isfinite(observed) & np.isfinite(predicted)
    if not mask.any():
        return None
    mse = float(np.mean((predicted[mask] - observed[mask]) ** 2))
    psnr = -10.0 * math.log10(mse) if mse > 0 else math.inf
    ssim = math.nan
    if min(observed.shape) >= _SSIM_MIN_SIDE:
        _, ssim_map = structural_similarity(
            np.where(mask, observed, 0.0),
            np.where(mask, predicted, 0.0),
            data_range=1.0,
            full=True,
        )
        ssim = float(np.mean(ssim_map[mask]))
    return {"loss": mse, "psnr": psnr, "ssim": ssim}


def _channel_cmap(channel):
    try:
        return plt.get_cmap(channel.get("cmap", "gray"))
    except ValueError:
        return plt.get_cmap("gray")


class EUVTomographyCallback(BaseCallback):
    """Log a configurable set of EUV reconstruction validation figures."""

    def __init__(
        self,
        ds_key,
        instrument_key,
        image_shape,
        channel_metadata,
        selected_channels,
        products,
        *,
        every_n_validations=1,
        figure_dpi=150,
        normalize=False,
    ):
        super().__init__(ds_key)
        self.instrument_key = str(instrument_key)
        self.normalize = bool(normalize)
        self.image_shape = tuple(int(value) for value in image_shape)
        self.products = products
        self.every_n_validations = int(every_n_validations)
        self.figure_dpi = int(figure_dpi)
        self._validation_count = 0

        channel_metadata = tuple(dict(channel) for channel in channel_metadata)
        indices = []
        selected_metadata = []
        for requested in selected_channels:
            matches = [
                (index, channel)
                for index, channel in enumerate(channel_metadata)
                if canonical_channel_id(channel["id"]) == canonical_channel_id(requested)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Callback channel {requested!r} must match exactly one artifact "
                    f"channel for {self.instrument_key!r}."
                )
            index, channel = matches[0]
            indices.append(index)
            selected_metadata.append(channel)
        self.channel_indices = tuple(indices)
        self.channel_metadata = tuple(selected_metadata)
        self.channel_ids = tuple(str(channel["id"]) for channel in channel_metadata)

        # Images are always needed for the validation metrics.
        requirements = {"valid_mask", "target_image", "pred_image"}
        if products["plasma_diagnostics"]["enabled"]:
            quantities = products["plasma_diagnostics"]["quantities"]
            requirements.update(_DIAGNOSTIC_FIELDS[quantity][0] for quantity in quantities)
            if any(quantity in _EMISSION_WEIGHTED_DIAGNOSTICS for quantity in quantities):
                requirements.add(_DIAGNOSTIC_FIELDS["emission_measure"][0])
        if products["thermal_distribution"]["enabled"]:
            requirements.add("differential_emission_measure_cm5_per_dex")
        if products["ray_sampling"]["enabled"]:
            requirements.update({"z_vals_stratified", "z_vals_hierarchical", "distance"})
        self.validation_output_keys = tuple(sorted(requirements))

    @property
    def state_key(self):
        return f"{type(self).__qualname__}:{self.ds_key}"

    def state_dict(self):
        return {"validation_count": self._validation_count}

    def load_state_dict(self, state_dict):
        self._validation_count = int(state_dict.get("validation_count", 0))

    def _reshape_outputs(self, outputs):
        reshaped = {}
        expected_pixels = math.prod(self.image_shape)
        for key, value in outputs.items():
            array = value.detach().cpu().numpy()
            if array.ndim == 0 or array.shape[0] != expected_pixels:
                raise ValueError(
                    f"Validation output {key!r} has shape {array.shape}; expected "
                    f"a leading image dimension of {expected_pixels}."
                )
            reshaped[key] = array.reshape(*self.image_shape, *array.shape[1:])
        return reshaped

    def _log_figure(self, key, figure):
        wandb.log({f"euv_tomography.{self.ds_key}.{key}": wandb.Image(figure)})
        plt.close(figure)

    def _plot_channel_comparison(self, outputs):
        settings = self.products["channel_comparison"]
        target = outputs["target_image"]
        prediction = outputs["pred_image"]
        valid_mask = outputs["valid_mask"].astype(bool)
        if valid_mask.shape != target.shape:
            valid_mask = np.broadcast_to(valid_mask, target.shape)
        rows = settings["rows"]
        figure, axes = plt.subplots(
            len(rows),
            len(self.channel_indices),
            figsize=(3.2 * len(self.channel_indices), 3.0 * len(rows)),
            dpi=self.figure_dpi,
            squeeze=False,
        )
        for column, (channel_index, metadata) in enumerate(
            zip(self.channel_indices, self.channel_metadata)
        ):
            channel_mask = valid_mask[..., channel_index]
            observed = np.where(channel_mask, target[..., channel_index], np.nan)
            predicted = np.where(channel_mask, prediction[..., channel_index], np.nan)
            if self.normalize:
                predicted = _match_mean_and_std(predicted, observed)
            # Both images arrive in loss space (the instrument scaling already
            # applied), so they are shown on a shared linear scale exactly as
            # the training loss compares them.
            intensity_norm = _linear_norm(
                np.concatenate([observed.ravel(), predicted.ravel()]),
                percentile=settings["intensity_percentile"],
            )
            residual = predicted - observed
            finite_residual = _finite_values(np.abs(residual))
            residual_limit = (
                float(np.nanpercentile(finite_residual, settings["residual_percentile"]))
                if finite_residual.size else 1.0
            )
            residual_limit = max(residual_limit, np.finfo(np.float32).eps)
            residual_norm = TwoSlopeNorm(
                vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
            )
            relative_floor = max(float(intensity_norm.vmax) * 1e-3, np.finfo(np.float32).eps)
            relative_residual = residual / np.maximum(np.abs(observed), relative_floor)
            finite_relative = _finite_values(np.abs(relative_residual))
            relative_limit = (
                float(np.nanpercentile(finite_relative, settings["residual_percentile"]))
                if finite_relative.size else 1.0
            )
            relative_limit = max(relative_limit, np.finfo(np.float32).eps)

            row_values = {
                "observation": (observed, _channel_cmap(metadata), intensity_norm),
                "prediction": (predicted, _channel_cmap(metadata), intensity_norm),
                "residual": (residual, "RdBu_r", residual_norm),
                "relative_residual": (
                    relative_residual,
                    "RdBu_r",
                    TwoSlopeNorm(vmin=-relative_limit, vcenter=0.0, vmax=relative_limit),
                ),
            }
            for row_index, row_name in enumerate(rows):
                values, cmap, norm = row_values[row_name]
                axis = axes[row_index, column]
                image = axis.imshow(values, origin="lower", cmap=cmap, norm=norm)
                title = f"{metadata['id']} — {row_name.replace('_', ' ')}"
                if self.normalize and row_name != "observation":
                    title += " (normalized)"
                axis.set_title(title)
                axis.set_axis_off()
                _append_colorbar(figure, axis, image)
        figure.tight_layout()
        return figure

    def _plot_plasma_diagnostics(self, outputs, renderer):
        settings = self.products["plasma_diagnostics"]
        quantities = _available_diagnostics(settings["quantities"], outputs, renderer)
        if not quantities:
            return None
        valid_mask = outputs["valid_mask"].astype(bool)
        image_mask = np.any(valid_mask, axis=-1) if valid_mask.ndim == 3 else valid_mask
        emission_mask = image_mask
        if any(quantity in _EMISSION_WEIGHTED_DIAGNOSTICS for quantity in quantities):
            emission_mask = _emission_measure_mask(
                np.squeeze(outputs[_DIAGNOSTIC_FIELDS["emission_measure"][0]]), image_mask
            )
        figure, axes = plt.subplots(
            1,
            len(quantities),
            figsize=(4.0 * len(quantities), 3.8),
            dpi=self.figure_dpi,
            squeeze=False,
        )
        for axis, quantity in zip(axes[0], quantities):
            field, title, cmap, norm_kind = _DIAGNOSTIC_FIELDS[quantity]
            values = np.squeeze(outputs[field])
            mask = emission_mask if quantity in _EMISSION_WEIGHTED_DIAGNOSTICS else image_mask
            values = np.where(mask, values, np.nan)
            if norm_kind == "log":
                norm = _positive_log_norm(values)
            elif norm_kind == "fraction":
                norm = Normalize(vmin=0.0, vmax=1.0)
            else:
                norm = _linear_norm(values)
            image = axis.imshow(values, origin="lower", cmap=cmap, norm=norm)
            axis.set_title(title)
            axis.set_axis_off()
            _append_colorbar(figure, axis, image)
        figure.tight_layout()
        return figure

    def _plot_thermal_distribution(self, outputs, renderer):
        settings = self.products["thermal_distribution"]
        dem = outputs["differential_emission_measure_cm5_per_dex"]
        valid_mask = outputs["valid_mask"].astype(bool)
        image_mask = np.any(valid_mask, axis=-1) if valid_mask.ndim == 3 else valid_mask
        samples = dem[image_mask]
        finite_rows = np.isfinite(samples).all(axis=-1) & np.any(samples > 0, axis=-1)
        samples = samples[finite_rows]
        log_temperature = renderer.log_T.detach().cpu().numpy()

        figure, axis = plt.subplots(figsize=(7, 4.5), dpi=self.figure_dpi)
        if samples.size:
            if settings["spatial_statistic"] == "median":
                center = np.nanmedian(samples, axis=0)
            else:
                center = np.nanmean(samples, axis=0)
            lower, upper = np.nanpercentile(
                samples, settings["percentile_band"], axis=0
            )
            positive = samples[samples > 0]
            floor = max(float(np.nanmin(positive)), np.finfo(np.float32).tiny)
            axis.fill_between(
                log_temperature,
                np.maximum(lower, floor),
                np.maximum(upper, floor),
                color="tab:orange",
                alpha=0.25,
                label=(
                    f"{settings['percentile_band'][0]:g}–"
                    f"{settings['percentile_band'][1]:g}%"
                ),
            )
            axis.plot(
                log_temperature,
                np.maximum(center, floor),
                color="tab:orange",
                label=settings["spatial_statistic"],
            )
            axis.set_yscale("log")
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No positive valid DEM samples", ha="center", va="center")
        axis.set_xlabel("log$_{10}$(T / K)")
        axis.set_ylabel("Differential emission measure [cm$^{-5}$ dex$^{-1}$]")
        axis.set_title(f"Thermal distribution — {self.instrument_key}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        return figure

    def _plot_response_and_gains(self, renderer):
        response = renderer.temperature_response.detach().cpu().numpy()
        # The response keeps its native temperature axis, not the diagnostic one.
        log_temperature = renderer.response_log_T.detach().cpu().numpy()
        density_index = response.shape[0] // 2
        response = response[density_index]
        channels = tuple(str(channel) for channel in renderer.channels)
        channel_gain_delta = (
            renderer.instrument_gain_delta_dex.detach().cpu().numpy()
        )
        common_gain = getattr(renderer, "common_gain_delta_dex", None)
        common_gain_delta = (
            0.0
            if common_gain is None
            else float(common_gain.detach().cpu().item())
        )
        gain_delta = channel_gain_delta + common_gain_delta

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=self.figure_dpi)
        for channel_index, channel in enumerate(channels):
            values = response[:, channel_index]
            positive = values > 0
            if np.any(positive):
                axes[0].plot(log_temperature[positive], values[positive], label=channel)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("log$_{10}$(T / K)")
        axes[0].set_ylabel(str(renderer.response_unit))
        title = "Temperature responses"
        if renderer.log_density_axis.numel():
            density = renderer.log_density_axis[density_index].item()
            title += f" at log$_{{10}}$(n$_e$ / cm$^{{-3}}$)={density:.2f}"
        axes[0].set_title(title)
        axes[0].legend(fontsize="small")
        axes[0].grid(alpha=0.25)

        axes[1].bar(channels, gain_delta, color="tab:blue")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].axhline(
            common_gain_delta,
            color="tab:orange",
            linestyle="--",
            label=f"common={common_gain_delta:.3f} dex",
        )
        axes[1].set_ylabel("Total learned gain correction [dex]")
        axes[1].set_title("Relative instrument + channel calibration")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend(fontsize="small")
        figure.suptitle(f"{self.instrument_key} — response {renderer.response_id}", fontsize=10)
        figure.tight_layout()
        return figure

    def _plot_ray_sampling(self, outputs):
        fraction_y, fraction_x = self.products["ray_sampling"]["pixel_fraction"]
        y = min(round(fraction_y * (self.image_shape[0] - 1)), self.image_shape[0] - 1)
        x = min(round(fraction_x * (self.image_shape[1] - 1)), self.image_shape[1] - 1)
        stratified = np.squeeze(outputs["z_vals_stratified"][y, x])
        hierarchical = np.squeeze(outputs["z_vals_hierarchical"][y, x])
        radius = np.squeeze(outputs["distance"][y, x])

        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=self.figure_dpi)
        axes[0].scatter(stratified, np.ones_like(stratified), s=14, label="stratified")
        axes[0].scatter(hierarchical, np.zeros_like(hierarchical), s=14, label="final")
        axes[0].set_yticks([0, 1], labels=["final", "stratified"])
        axes[0].set_xlabel("Distance along ray [model units]")
        axes[0].set_title(f"Ray samples at pixel ({y}, {x})")
        axes[0].grid(alpha=0.25)
        axes[1].plot(hierarchical, radius, marker=".")
        axes[1].set_xlabel("Distance along ray [model units]")
        axes[1].set_ylabel(r"Heliocentric radius [$R_\odot$]")
        axes[1].set_title("Sampled shell trajectory")
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        return figure

    def _compute_metrics(self, outputs):
        """Loss, PSNR and SSIM per channel and averaged over all channels."""
        target = outputs["target_image"]
        prediction = outputs["pred_image"]
        valid_mask = outputs["valid_mask"].astype(bool)
        if valid_mask.shape != target.shape:
            valid_mask = np.broadcast_to(valid_mask, target.shape)
        metrics = {}
        per_channel = []
        for channel_index, channel_id in enumerate(self.channel_ids):
            mask = valid_mask[..., channel_index]
            observed = np.where(mask, target[..., channel_index], np.nan)
            predicted = np.where(mask, prediction[..., channel_index], np.nan)
            if self.normalize:
                predicted = _match_mean_and_std(predicted, observed)
            channel_metrics = _image_metrics(observed, predicted, mask)
            if channel_metrics is None:
                continue
            per_channel.append(channel_metrics)
            for name, value in channel_metrics.items():
                metrics[f"validation/{name}.{self.ds_key}.{channel_id}"] = value
        if per_channel:
            loss = float(np.mean([entry["loss"] for entry in per_channel]))
            metrics[f"validation/loss.{self.ds_key}"] = loss
            metrics[f"validation/psnr.{self.ds_key}"] = (
                -10.0 * math.log10(loss) if loss > 0 else math.inf
            )
            metrics[f"validation/ssim.{self.ds_key}"] = float(
                np.mean([entry["ssim"] for entry in per_channel])
            )
        return metrics

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        self._validation_count += 1
        plot_figures = self._validation_count % self.every_n_validations == 0
        raw_outputs = self.get_validation_outputs(pl_module)
        if raw_outputs is None:
            return
        outputs = self._reshape_outputs(raw_outputs)
        # Metrics are cheap and logged at every validation; figures follow
        # every_n_validations.
        metrics = self._compute_metrics(outputs)
        if metrics:
            wandb.log(metrics)
        if not plot_figures:
            return
        renderer = pl_module.rendering.rendering_modules[self.instrument_key]

        if self.products["channel_comparison"]["enabled"]:
            self._log_figure(
                "channel_comparison", self._plot_channel_comparison(outputs)
            )
        if self.products["plasma_diagnostics"]["enabled"]:
            figure = self._plot_plasma_diagnostics(outputs, renderer)
            if figure is not None:
                self._log_figure("plasma_diagnostics", figure)
        if self.products["thermal_distribution"]["enabled"]:
            self._log_figure(
                "thermal_distribution",
                self._plot_thermal_distribution(outputs, renderer),
            )
        if self.products["response_and_gains"]["enabled"]:
            self._log_figure(
                "response_and_gains", self._plot_response_and_gains(renderer)
            )
        if self.products["ray_sampling"]["enabled"]:
            self._log_figure("ray_sampling", self._plot_ray_sampling(outputs))
