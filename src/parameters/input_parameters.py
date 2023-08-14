
import json
from os.path import dirname, join
from typing import Callable, List

import numpy as np

from genetic_algorithm.genetic_algorithm import ga_map
from genetic_algorithm.genetic_problem import classic_fitness, fitness
from spectrum.experimental_spectrum import ExperimentalSpectrum
from spectrum.spectrum_type import string_to_spectrum_type


class InputParameters:
    def __init__(self, path: str):
        with open(path) as f:
            self.params = json.load(f)

        self.experimental_spectra = [
            ExperimentalSpectrum.from_path(
                path=join(dirname(path), spectrum_data["file"]),
                type=string_to_spectrum_type(spectrum_data["type"]),
                mirroring_option=spectrum_data["mirroring_option"],
                hwhm=spectrum_data["hwhm"],
                freq_range=tuple(spectrum_data["interval"]),
                scaling_factors=spectrum_data["scaling_factors"],
                is_opt_candidate=spectrum_data["optimise"],
                energies=self.energies,
            )
            for spectrum_data in self.params["spectra_data"]
        ]

    @property
    def energy_uncertainty(self):
        return self.params["energy_uncertainty"]

    @property
    def energies(self):
        return self.params["energies"]

    @property
    def eu(self):
        return self.params["energy_unit"]

    @property
    def skip_print(self):
        return self.params["skip_print"]

    @property
    def genetic_algorithm(self):
        if (ga := self.params.get("genetic_algorithm")) in ga_map:
            return ga
        else:
            valid_gas = ', '.join(ga_map.keys())
            raise KeyError(f'Invalid genetic algorithm "{ga}". Valid options are: {valid_gas}')

    @property
    def candidates(self):
        return [spectrum for spectrum in self.experimental_spectra if spectrum.is_opt_candidate]

    def energies_array(self) -> np.ndarray:
        return np.array(list(self.energies.values()), dtype=np.float64)

    def objective_function(self) -> Callable[[np.array], float]:
        def obj(x: np.array) -> float:
            return - np.prod([fitness(x, candidate, self.eu) for candidate in self.candidates])

        return obj

    def classic_objective_function(self) -> Callable[[np.array], float]:
        def obj(x: np.array) -> float:
            return - np.prod([classic_fitness(x, candidate, self.eu) for candidate in self.candidates])

        return obj
