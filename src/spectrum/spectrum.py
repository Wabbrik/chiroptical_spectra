from enum import Enum, auto
from os.path import join, dirname
from typing import Dict, List, Tuple

import numpy as np

from spectrum.broadening import broaden_delegate


class SpectrumType(Enum):
    VCD = auto()
    IR = auto()
    ECD = auto()
    UV = auto()


def string_to_spectrum_type(string: str) -> SpectrumType:
    if string == 'VCD':
        return SpectrumType.VCD
    if string == 'IR':
        return SpectrumType.IR
    if string == 'ECD':
        return SpectrumType.ECD
    if string == 'UV':
        return SpectrumType.UV
    raise ValueError(f'Unknown spectrum type: {string}')


class Spectrum:
    @classmethod
    def from_path(cls, path: str) -> "Spectrum":
        data = np.loadtxt(path)
        return cls(freq=data[:, 0], vals=data[:, 1])

    def __init__(self, freq: np.array, vals: np.array) -> None:
        self._freq = freq
        self._vals = vals

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


class ExperimentalSpectrum(Spectrum):
    @classmethod
    def from_path(
        cls,
        path: str,
        type: SpectrumType,
        mirroring_option: float,
        hwhm: float,
        freq_range: Tuple[float, float],
        scaling_factors: List[List[float]],
        is_opt_candidate: bool,
        energies: Dict[str, float]
    ) -> "ExperimentalSpectrum":
        data = np.loadtxt(path)
        return cls(
            freq=data[:, 0],
            vals=data[:, 1],
            broadening_dir=dirname(path),
            type=type,
            mirroring_option=mirroring_option,
            hwhm=hwhm,
            freq_range=freq_range,
            scaling_factors=scaling_factors,
            is_opt_candidate=is_opt_candidate,
            energies=energies
        )

    def __init__(
        self,
        freq: np.array,
        vals: np.array,
        broadening_dir: str,
        type: SpectrumType,
        mirroring_option: float,
        hwhm: float,
        freq_range: Tuple[float, float],
        scaling_factors: List[List[float]],
        is_opt_candidate: bool,
        energies: Dict[str, float]
    ) -> None:
        super().__init__(freq, vals)
        self.type = type
        self.mirroring_option = mirroring_option
        self.hwhm = hwhm
        self.freq_range = freq_range
        self.scaling_factors = scaling_factors
        self.is_opt_candidate = is_opt_candidate
        self.energies = energies
        self.broadened: Dict[str, Spectrum] = self._broaden(
            broadening_dir, self.energies
        )

    def __str__(self) -> str:
        return f"""ExperimentalSpectrum(
                    freq={self._freq},
                    vals={self._vals},
                    type={self.type},
                    hwhm={self.hwhm},
                    freq_range={self.freq_range},
                    scaling_factors={self.scaling_factors},
                    optimise={self.is_opt_candidate},
                    energies={self.energies}
                )"""

    def energies_array(self) -> np.ndarray:
        return np.array(list(self.energies.values()))

    def write_broadened(self, dir: str) -> None:
        for fname, spectrum in self.broadened.items():
            spectrum.write(join(dir, f"{fname}.dat"))

    def _broaden(self, dirpath: str, energies: Dict[str, float]) -> Dict[str, Spectrum]:
        return {
            fname: broaden_delegate(
                s=Spectrum.from_path(join(dirpath, fname)), es=self
            )
            for fname in list(energies.keys())
        }
