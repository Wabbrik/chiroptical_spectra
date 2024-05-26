from os.path import dirname, join
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from broadening.broadening import ecd_broaden, ir_broaden, uv_broaden, vcd_broaden
from spectrum.spectrum import Spectrum
from spectrum.spectrum_type import SpectrumType, prefix_by_type


class ExperimentalSpectrum(Spectrum):
    @classmethod
    def from_path(
        cls,
        path: str,
        type: SpectrumType,
        mirroring_option: float,
        path_length: float,
        molar_concentration: float,
        hwhm: float,
        freq_range: Tuple[float, float],
        scaling_factors: List[List[float]],
        is_opt_candidate: bool,
        is_reference_candidate: bool,
        energies: Dict[str, float],
    ) -> "ExperimentalSpectrum":
        data = np.loadtxt(path, dtype=np.float64)
        return cls(
            freq=data[:, 0],
            vals=data[:, 1],
            broadening_dir=dirname(path),
            type=type,
            mirroring_option=mirroring_option,
            path_length=path_length,
            molar_concentration=molar_concentration,
            hwhm=hwhm,
            freq_range=freq_range,
            scaling_factors=scaling_factors,
            is_opt_candidate=is_opt_candidate,
            is_reference_candidate=is_reference_candidate,
            energies=energies,
        )

    def __init__(
        self,
        freq: npt.NDArray[np.float64],
        vals: npt.NDArray[np.float64],
        broadening_dir: str,
        type: SpectrumType,
        mirroring_option: float,
        path_length: float,
        molar_concentration: float,
        hwhm: float,
        freq_range: Tuple[float, float],
        scaling_factors: List[List[float]],
        is_opt_candidate: bool,
        is_reference_candidate: bool,
        energies: Dict[str, float],
    ) -> None:
        super().__init__(freq, vals)
        self.type = type
        self.mirroring_option = mirroring_option
        self.path_length = path_length
        self.molar_concentration = molar_concentration
        self.hwhm = hwhm
        self.freq_range = freq_range
        self.scaling_factors = scaling_factors
        self.is_opt_candidate = is_opt_candidate
        self.is_reference_candidate = is_reference_candidate
        self.energies = energies
        self.broadened: Dict[str, Spectrum] = self._broaden(broadening_dir, energies)
        self.broadened_vals = np.array([spec.vals(self.freq_range) for spec in self.broadened.values()])

    def write_broadened(self, dir: str) -> None:
        for fname, spectrum in self.broadened.items():
            spectrum.write(join(dir, f"{fname}.dat"))

    def _broaden(self, dirpath: str, energies: Dict[str, float]) -> Dict[str, Spectrum]:
        broaden_funcs = {
            SpectrumType.ROA: ir_broaden,
            SpectrumType.VCD: vcd_broaden,
            SpectrumType.IR: ir_broaden,
            SpectrumType.ECD: ecd_broaden,
            SpectrumType.UV: uv_broaden,
        }

        return {
            fname: broaden_funcs[self.type](
                spectrum=Spectrum.from_path(join(dirpath, (f"{prefix_by_type[self.type]}{fname}"))),
                freq_range=self.freq_range,
                hwhm=self.hwhm,
                grid=self.freq(),
                intervals=self.scaling_factors,
            )
            * self.mirroring_option
            * (1 / (self.path_length * self.molar_concentration))
            for fname in energies
        }

    def simulated_vals(self, weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return weights @ self.broadened_vals
