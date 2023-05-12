from typing import Callable, List

import numpy as np
from pymoo.problems.functional import FunctionalProblem

from overlap.metrics import tanimoto
from overlap.weights import boltzmann_weights
from spectrum.spectrum import ExperimentalSpectrum


def fitness(x_energies: np.array[float], es: ExperimentalSpectrum, constant: str = "kcal/mol") -> np.array:
    weights = boltzmann_weights(x_energies, constant=constant)
    broadened = list(es.broadened.values())
    spectra_accumulator = np.sum(
        [weights[i] * spec.vals(es.freq_range) for i, spec in enumerate(broadened)])
    return -1 * tanimoto(spectra_accumulator, es.vals(range))


class GeneticProblem(FunctionalProblem):
    def __init__(self, energies: dict, error: float, objs: Callable[[np.array], float]):
        x_l, x_u = energies - error, energies + error
        super().__init__(n_var=len(energies), objs=objs, xl=x_l, xu=x_u, type_var=float)
