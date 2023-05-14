from typing import Callable, List

import numpy as np
from pymoo.problems.functional import FunctionalProblem

from overlap.metrics import tanimoto
from overlap.weights import boltzmann_weights
from spectrum.experimental_spectrum import ExperimentalSpectrum


def fitness(x_energies: np.ndarray, es: ExperimentalSpectrum, constant: str = "kcal/mol") -> np.array:
    weights = boltzmann_weights(x_energies, constant=constant)
    broadened = list(es.broadened.values())
    spectra_accumulator = np.sum(
        [weights[i] * spec.vals(es.freq_range) for i, spec in enumerate(broadened)], axis=0)
    return tanimoto(spectra_accumulator, es.vals(es.freq_range))


class GeneticProblem(FunctionalProblem):
    def __init__(self, energies: dict, error: float, objs: Callable[[np.ndarray], float]):
        x_l, x_u = energies - error, energies + error
        super().__init__(n_var=len(energies), objs=objs, xl=x_l, xu=x_u, type_var=float)
