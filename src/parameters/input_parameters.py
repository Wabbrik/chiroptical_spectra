
import json
from os.path import dirname, join
from typing import Callable, Dict, List

import numpy as np

from genetic_algorithm.genetic_algorithm import ga_map
from genetic_algorithm.genetic_problem import fitness
from spectrum.broadening import broaden
from spectrum.spectrum import (ExperimentalSpectrum, Spectrum,
                               string_to_spectrum_type)


class InputParameters:
    def __init__(self, path: str):
        self.base_path = dirname(path)
        with open(path) as f:
            self.params = json.load(f)
        self._set_params(dirname(path))

    def __assert_preconditions(self) -> None:
        # should probably validate the schema in some way
        ...

    def _set_params(self) -> None:
        self.__assert_preconditions()
        self.energy_uncertainty = self.params["energy_uncertainty"]
        self.experimental_spectra = self._get_experimental_spectra()
        self.genetic_algorithm = self._get_genetic_algorithm()
        self.skip_print, self.energy_unit = self.params["skip_print"], self.params["energy_unit"]
        self.energies = self._get_energies()
        self.broadened_spectra = self._broaden_spectra()

    def _get_genetic_algorithm(self) -> str:
        if ga := self.params["genetic_algorithm"] not in ga_map.keys():
            raise KeyError(
                f'Genetic algorithm {ga} not found, try one of: {", ".join(ga_map.keys())}'
            )
        return ga

    def _get_experimental_spectra(self) -> Dict[str, ExperimentalSpectrum]:
        experimental_spectra = {}
        for spectrum, info in self.params["experimental_spectra"]:
            experimental_spectra[spectrum] = ExperimentalSpectrum.from_path(
                path=join(self.base_path, info["file"]),
                type=string_to_spectrum_type(info["type"]),
                mirroring_option=info["mirroring_option"],
                hwhm=info["hwhm"],
                freq_range=info["interval"],
                scaling_factors=info["scaling_factors"]
            )
        return experimental_spectra

    def optimisation_candidates(self) -> Dict[str, ExperimentalSpectrum]:
        return {  # used to determine objectives
            spectrum: self.experimental_spectra[spectrum]
            for spectrum, info in self.params["experimental_spectra"]
            if info["optimise"] == True
        }

    def _get_energies(self) -> Dict[str, Dict[str, float]]:
        def assert_same_energies(mappings: List[Dict[str, float]]):
            if mappings and len(mappings) > 1:
                arrays = [np.array(list(mapping.values()))
                          for mapping in mappings]
                if not all(np.array_equal(arrays[0], arr) for arr in arrays[1:]):
                    raise ValueError("Mismatched energies")

        assert_same_energies(list(self.params["energies"].values()))
        return self.params["energies"].items()

    def energies_array(self) -> np.ndarray:
        return np.array(
            list(
                self.energies[next(iter(self.experimental_spectra))]
                 .values()  # should be the same regardless of experiment
            )
        )

    def _broaden_spectra(self) -> Dict[str, Dict[str, Spectrum]]:
        broadend_spectra = {}
        for exp_spectrum, spectrum_mapping in self.energies.items():
            broadend_spectra[exp_spectrum] = {
                fname: broaden(s=Spectrum.from_path(
                    join(self.base_path, fname)), es=self.experimental_spectra[exp_spectrum])
                for fname, _ in spectrum_mapping.items()}

        return broadend_spectra

    def objective_function(self) -> Callable[[np.array], float]:
        def obj(x: np.array) -> float:
            return np.prod([fitness(x, list(self.broadened_spectra[experiment].values()), candidate)
                            for experiment, candidate in self.optimisation_candidates().items()])
        return obj
