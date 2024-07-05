import os
from os.path import dirname, join
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from spectrum.spectrum import Spectrum
from spectrum.spectrum_type import SpectrumType, broaden_funcs, prefix_by_type


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
        already_broadened: bool,
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
            already_broadened=already_broadened,
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
        already_broadened: bool,
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
        self.broadened, self.broadened_vals = self._broadening(broadening_dir, energies, already_broadened)

    def _broadening(
        self, broadening_dir: str, energies: Dict[str, float], already_broadened: bool
    ) -> Tuple[Dict[str, Spectrum], npt.NDArray[np.float64]] | None:
        broadened = (
            self.__broaden(broadening_dir, energies)
            if not already_broadened
            else self.__skip_broaden(broadening_dir, energies)
        )
        try:
            return broadened, np.array([spec.vals(self.freq_range) for spec in broadened.values()])
        except ValueError:
            print(
                f"There was an issue with the broadened spectra. Are you sure the broadening is correct? already_broadened={already_broadened}"
            )
            raise

    def __write_broadened(self, broadened: Dict[str, Spectrum], dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        for fname, spectrum in broadened.items():
            spectrum.write(join(dir, f"{prefix_by_type[self.type]}{fname}.dat"))

    def __broaden(self, dirpath: str, energies: Dict[str, float]) -> Dict[str, Spectrum]:
        broadened = {
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
        self.__write_broadened(broadened, f"{dirpath}/Spectra_{self.type.name}")
        return broadened

    def __skip_broaden(self, dirpath: str, energies: Dict[str, float]) -> Dict[str, Spectrum]:
        return {fname: Spectrum.from_path(join(dirpath, (f"{prefix_by_type[self.type]}{fname}"))) for fname in energies}

    def simulated_vals(self, weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return weights @ self.broadened_vals
