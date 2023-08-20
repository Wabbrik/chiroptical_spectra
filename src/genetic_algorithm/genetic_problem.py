from typing import Callable

import numpy as np
import numpy.typing as npt
from pymoo.problems.functional import FunctionalProblem

from overlap.metrics import tanimoto, fitness_tanimoto
from overlap.weights import boltzmann_weights
from spectrum.experimental_spectrum import ExperimentalSpectrum

_rng = np.random.default_rng(seed=123)


def fitness(
    x_energies: npt.NDArray[np.float64],
    es: ExperimentalSpectrum,
    constant: str = "kcal/mol",
    dropout_percentage: float = 0.08
) -> npt.NDArray[np.float64]:
    weights = boltzmann_weights(x_energies, constant)
    weights[_rng.random(len(weights)) < dropout_percentage] = 0.0
    spectra_accumulator = es.simulated_vals(weights)
    return tanimoto(spectra_accumulator, es.vals(es.freq_range))


def classic_fitness(
    x_energies: npt.NDArray[np.float64],
    es: ExperimentalSpectrum,
    constant: str = "kcal/mol",
) -> npt.NDArray[np.float64]:
    spectra_accumulator = es.simulated_vals(boltzmann_weights(x_energies, constant))
    return fitness_tanimoto(spectra_accumulator, es.vals(es.freq_range))


class GeneticProblem(FunctionalProblem):
    def __init__(self, chromosome: npt.NDArray[np.float64], error: float, objs: Callable[[npt.NDArray[np.float64]], float]):
        x_l, x_u = chromosome - error, chromosome + error
        super().__init__(n_var=len(chromosome), objs=objs, xl=x_l, xu=x_u, type_var=float)
