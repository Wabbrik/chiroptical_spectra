from typing import Callable, List

import numpy as np
from pymoo.problems.functional import FunctionalProblem

from overlap.metrics import tanimoto
from overlap.weights import boltzmann_weights
from spectrum.spectrum import ExperimentalSpectrum, Spectrum


def fitness(x_energies: np.array[float], broadend: List[Spectrum], experimental_spectrum: ExperimentalSpectrum) -> np.array:
    weights = boltzmann_weights(x_energies)
    range = experimental_spectrum.freq_range
    spectra_accumulator = np.sum([weights[i] * spec.vals(range)
                                  for i, spec in enumerate(broadend)])
    return -1 * tanimoto(spectra_accumulator, experimental_spectrum.vals(range))


class GeneticProblem(FunctionalProblem):
    def __init__(self, energies: dict, error: float, objs: Callable[[np.array], float]):
        x_l, x_u = energies - error, energies + error
        super().__init__(n_var=len(energies), objs=objs, xl=x_l, xu=x_u, type_var=float)
