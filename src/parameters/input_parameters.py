import json
from os.path import dirname, join
from typing import Dict, List

import numpy as np
import numpy.typing as npt

from genetic_algorithm.genetic_algorithm import ga_map
from spectrum.experimental_spectrum import ExperimentalSpectrum
from spectrum.spectrum_type import string_to_spectrum_type


class InputParameters:
    def __init__(self, path: str) -> None:
        with open(path) as f:
            self.params = json.load(f)

        self.experimental_spectra = [
            ExperimentalSpectrum.from_path(
                path=join(dirname(path), spectrum_data["file"]),
                type=string_to_spectrum_type(spectrum_data["type"]),
                mirroring_option=spectrum_data["mirroring_option"],
                path_length=spectrum_data["path_length"],
                molar_concentration=spectrum_data["molar_concentration"],
                hwhm=spectrum_data["hwhm"],
                freq_range=tuple(spectrum_data["interval"]),
                scaling_factors=spectrum_data["scaling_factors"],
                is_opt_candidate=spectrum_data["optimise"],
                energies=self.energies,
            )
            for spectrum_data in self.params["spectra_data"]
        ]

    @property
    def termination_criterion_ngen(self) -> float:
        return self.params["termination_criterion(ngen)"]

    @property
    def objective(self) -> str:
        return self.params["objective"]

    @property
    def energy_uncertainty(self) -> float:
        return self.params["energy_uncertainty"]

    @property
    def energies(self) -> Dict[str, float]:
        return self.params["energies"]

    @property
    def eu(self) -> str:
        return self.params["energy_unit"]

    @property
    def skip_print(self) -> bool:
        return self.params["skip_print"]

    @property
    def genetic_algorithm(self) -> str:
        if (ga := self.params.get("genetic_algorithm")) in ga_map:
            return ga
        else:
            raise KeyError(f'Invalid genetic algorithm "{ga}". Valid options are: {list(ga_map)}')

    @property
    def candidates(self) -> List[ExperimentalSpectrum]:
        return [spectrum for spectrum in self.experimental_spectra if spectrum.is_opt_candidate]

    def energies_array(self) -> npt.NDArray[np.float64]:
        return np.array(list(self.energies.values()), dtype=np.float64)
