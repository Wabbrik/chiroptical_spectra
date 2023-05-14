from os.path import dirname, join
from typing import Dict, List, Tuple

import numpy as np

from broadening.broadening import (ecd_broaden, ir_broaden, uv_broaden,
                                   vcd_broaden)
from spectrum.spectrum import Spectrum
from spectrum.spectrum_type import SpectrumType


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
        data = np.loadtxt(path, dtype=np.float32)
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
        return np.array(list(self.energies.values()), dtype=np.float32)

    def write_broadened(self, dir: str) -> None:
        for fname, spectrum in self.broadened.items():
            spectrum.write(join(dir, f"{fname}.dat"))

    def _broaden(self, dirpath: str, energies: Dict[str, float]) -> Dict[str, Spectrum]:
        return {
            fname: self.broaden_delegate(
                s=Spectrum.from_path(join(dirpath, fname))
            )
            for fname in list(energies.keys())
        }

    def broaden_delegate(self, s: Spectrum) -> Spectrum:
        broaden_funcs = {
            SpectrumType.VCD: vcd_broaden,
            SpectrumType.IR: ir_broaden,
            SpectrumType.ECD: ecd_broaden,
            SpectrumType.UV: uv_broaden,
        }

        broaden_func = broaden_funcs.get(self.type)
        if broaden_func is None:
            raise ValueError(f"Unknown spectrum type: {self.type}")

        return broaden_func(
            spectrum=s, freq_range=self.freq_range, hwhm=self.hwhm, grid=self.freq(), intervals=self.scaling_factors
        ) * self.mirroring_option
