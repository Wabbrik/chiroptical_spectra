from ast import List
from math import sqrt
from typing import Tuple

import numpy as np

from spectrum.spectrum import Spectrum


def get_scale_factor(freq_value: float, scaling_factors: list) -> float:
    """Given a list of shape ((1, 3), n) where [lower, upper, scale_value]"""
    return next(
        (freq_value * scale for lower, upper, scale in scaling_factors if lower < freq_value <= upper),
        freq_value,
    )


def vcd_broaden(
        spectrum: Spectrum,
        freq_range: Tuple[float, float],
        hwhm: float,
        grid: np.ndarray,
        intervals: List
) -> Spectrum:
    new_x, rs_y = grid.astype(dtype=np.single), np.zeros(grid.shape, dtype=np.single)
    lower, upper = sorted(freq_range)
    hwhm_ = 15 * hwhm
    t_hwhm_2 = np.full([len(new_x), ], hwhm ** 2, dtype=np.single)
    t_x229600 = (new_x / 229600).astype(np.single)

    for i, x_value in enumerate(spectrum.freq()):
        if (lower - hwhm_) < x_value < (upper + hwhm_):
            t_y = np.full([len(new_x), ], spectrum.vals()[i] / np.pi, dtype=np.single)
            t_sf = np.full([len(new_x), ], get_scale_factor(x_value, intervals), dtype=np.single)
            rs_y += (t_y * t_x229600 * hwhm / ((new_x - t_sf) ** 2 + t_hwhm_2))

    return Spectrum(new_x, rs_y)


def ir_broaden(
        spectrum: Spectrum,
        freq_range: Tuple[float, float],
        hwhm: float,
        grid: np.ndarray,
        intervals: List
) -> Spectrum:
    new_x, ds_y = grid.astype(dtype=np.single), np.zeros(grid.shape, dtype=np.single)
    lower, upper = sorted(freq_range)
    hwhm_ = 15 * hwhm
    t_hwhm_2 = np.full([len(new_x), ], hwhm ** 2, dtype=np.single)
    t_x9184 = (new_x / 91.84).astype(np.single)

    for i, x_value in enumerate(spectrum.freq()):
        if (lower - hwhm_) < x_value < (upper + hwhm_):
            t_y = np.full([len(new_x), ], (spectrum.vals()[i] / np.pi), dtype=np.single)
            t_sf = np.full([len(new_x), ], get_scale_factor(x_value, intervals), dtype=np.single)
            ds_y += t_y * t_x9184 * hwhm / ((new_x - t_sf) ** 2 + t_hwhm_2)

    return Spectrum(new_x, ds_y)


def ecd_broaden(
        spectrum: Spectrum,
        hwhm: float,
        grid: np.ndarray,
        intervals: List,
        **kwargs,
) -> Spectrum:
    new_x, ecd_y = grid.astype(dtype=np.single), np.zeros(grid.shape, dtype=np.single)
    rcp_hwhm = 1.0 / hwhm
    epsilon_constant = 1.0 / (22.94 * hwhm * sqrt(np.pi))

    for i, x_value in enumerate(spectrum.freq()):
        energy_nm = np.full([len(new_x), ], get_scale_factor(x_value, intervals), dtype=np.single)
        ecd_delta_epsilon = energy_nm * spectrum.vals()[i] * epsilon_constant
        ecd_y += ecd_delta_epsilon * np.exp(-((new_x - energy_nm) * rcp_hwhm) ** 2)

    return Spectrum(new_x, ecd_y)


def uv_broaden(
    spectrum: Spectrum,
    hwhm: float,
    grid: np.ndarray,
    intervals: List,
    **kwargs,
) -> Spectrum:
    new_x, uv_y = grid.astype(dtype=np.single), np.zeros(grid.shape, dtype=np.single)

    for i, x_value in enumerate(spectrum.freq()):
        energy_nm = np.full([len(new_x), ], get_scale_factor(x_value, intervals), dtype=np.single)
        uv_epsilon = 13.064 * energy_nm * energy_nm * spectrum.vals()[i] / hwhm
        uv_y += uv_epsilon * np.exp(-((new_x - energy_nm) / hwhm) ** 2)

    return Spectrum(new_x, uv_y)
