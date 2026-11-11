"""Extract band-power node features."""

from __future__ import annotations

import numpy as np


def band_power(
    window: np.ndarray,
    sampling_rate_hz: float,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(window.shape[-1], 1 / sampling_rate_hz)
    spectrum = np.abs(np.fft.rfft(window, axis=-1)) ** 2
    values = []
    total_power = spectrum.sum(axis=-1, keepdims=True)
    for low, high in bands.values():
        mask = (frequencies >= low) & (frequencies < high)
        values.append(spectrum[..., mask].mean(axis=-1))
    power = np.stack(values, axis=-1)
    return np.log(np.maximum(power / np.maximum(total_power, 1e-12), 1e-12)).astype(
        np.float32
    )
