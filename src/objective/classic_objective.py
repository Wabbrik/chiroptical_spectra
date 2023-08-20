from typing import Callable, List

import numpy as np
import numpy.typing as npt

from objective.objective import Objective
from spectrum.experimental_spectrum import ExperimentalSpectrum


class ClassicObjective(Objective):
    def __init__(
        self,
        energies_array: npt.NDArray[np.float64],
        candidates: List[ExperimentalSpectrum],
        error: float,
        energy_unit: str,
        fitness_function: Callable[[npt.NDArray[np.float64]], float],
    ) -> None:
        self._energies_array = energies_array
        self._optimization_candidates = candidates
        self._error = error
        self._eu = energy_unit
        self.fitness = fitness_function

    @property
    def lower(self) -> npt.NDArray[np.float64]:
        return self._energies_array - self._error

    @property
    def upper(self) -> npt.NDArray[np.float64]:
        return self._energies_array + self._error

    @property
    def n_var(self) -> int:
        return self._energies_array.size

    def process(self, x: npt.NDArray[np.float64]) -> np.float64:
        return -np.prod([self.fitness(x, candidate, self._eu) for candidate in self._optimization_candidates])
