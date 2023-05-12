
import json
from os.path import dirname, join
from typing import Callable, List

import numpy as np

from genetic_algorithm.genetic_algorithm import ga_map
from genetic_algorithm.genetic_problem import fitness
from spectrum.spectrum import (ExperimentalSpectrum,
                               string_to_spectrum_type)


class InputParameters:
    def __init__(self, path: str):
        self.base_path = dirname(path)
        with open(path) as f:
            self.params = json.load(f)
        self._set_params()

    def __assert_preconditions(self) -> None:
        # should probably validate the schema in some way
        ...

    def __assert_postconditions(self) -> None:
        # energy values should be the same and in the same order
        energies = [spectrum.energies_array()
                    for spectrum in self.experimental_spectra]
        if len(energies) > 1:
            if not all(np.array_equal(energies[0], arr) for arr in energies[1:]):
                raise ValueError("Given energies were not the same.")

    def _set_params(self) -> None:
        self.__assert_preconditions()
        self.energy_uncertainty = self.params["energy_uncertainty"]
        self.genetic_algorithm = self._get_genetic_algorithm()
        self.skip_print, self.energy_unit = self.params["skip_print"], self.params["energy_unit"]
        self.experimental_spectra = self._get_experimental_spectra()
        self.__assert_postconditions()

    def _get_genetic_algorithm(self) -> str:
        if ga := self.params["genetic_algorithm"] not in ga_map.keys():
            raise KeyError(
                f'Genetic algorithm {ga} not found, try one of: {", ".join(ga_map.keys())}'
            )
        return ga

    def _get_experimental_spectra(self) -> List[ExperimentalSpectrum]:
        return [
            ExperimentalSpectrum.from_path(
                path=join(self.base_path, spectrum_data["file"]),
                type=string_to_spectrum_type(spectrum_data["type"]),
                mirroring_option=spectrum_data["mirroring_option"],
                hwhm=spectrum_data["hwhm"],
                freq_range=spectrum_data["interval"],
                scaling_factors=spectrum_data["scaling_factors"],
                is_opt_candidate=spectrum_data["optimise"],
                energies=spectrum_data["energies"]
            )
            for spectrum_data in self.params["spectra_data"]
        ]

    def energies_array(self) -> np.ndarray:
        return next(self.experimental_spectra).energies_array()

    def objective_function(self) -> Callable[[np.array], float]:
        candidates = [
            spectrum for spectrum in self.experimental_spectra if spectrum.is_opt_candidate]

        def obj(x: np.array) -> float:
            return np.prod([fitness(x, candidate, self.energy_unit) for candidate in candidates])

        return obj
