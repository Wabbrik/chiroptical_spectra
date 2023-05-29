from typing import Tuple

import numpy as np


class Spectrum:
    @classmethod
    def from_path(cls, path: str) -> "Spectrum":
        data = np.loadtxt(path, dtype=np.float64)
        return cls(freq=data[:, 0], vals=data[:, 1])

    def __init__(self, freq: np.array, vals: np.array) -> None:
        self._freq = freq.astype(np.float64)
        self._vals = vals.astype(np.float64)

    def __str__(self) -> str:
        return f'Spectrum(freq={self._freq}, vals={self._vals})'

    def __repr__(self) -> str:
        return self.__str__()

    def __mul__(self, other: float) -> "Spectrum":
        return Spectrum(self._freq, self._vals * other)

    def __rmul__(self, other: float) -> "Spectrum":
        return self.__mul__(other)

    def __imul__(self, other: float) -> "Spectrum":
        self._vals *= other
        return self

    def vals(self, freq_range: Tuple[float, float] = None) -> np.ndarray:
        if freq_range is None:
            return self._vals
        start, end = sorted(freq_range)
        return self._vals[(self._freq >= start) & (self._freq <= end)]

    def freq(self, freq_range: Tuple[float, float] = None) -> np.ndarray:
        if freq_range is None:
            return self._freq
        start, end = sorted(freq_range)
        return self._freq[(self._freq >= start) & (self._freq <= end)]

    def write(self, path: str) -> None:
        with open(path, mode='w', encoding="utf8") as file:
            file.write(
                '\n'.join(f'{x:<10.8f} {y:>10.8f}' for x, y in zip(self._freq, self._vals)))
