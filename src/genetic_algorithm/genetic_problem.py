from typing import Callable

import numpy as np
from pymoo.problems.functional import FunctionalProblem

from overlap.metrics import tanimoto, fitness_tanimoto
from overlap.weights import boltzmann_weights
from spectrum.experimental_spectrum import ExperimentalSpectrum

_rng = np.random.default_rng(seed=123)


def fitness(
    x_energies: np.ndarray,
    es: ExperimentalSpectrum,
    constant: str = "kcal/mol",
    dropout_percentage: float = 0.08
) -> np.array:
    weights = boltzmann_weights(x_energies, constant)
    weights[_rng.random(len(weights)) < dropout_percentage] = 0.0
    spectra_accumulator = es.simulated_vals(weights)
    return tanimoto(spectra_accumulator, es.vals(es.freq_range))


def classic_fitness(
    x_energies: np.ndarray,
    es: ExperimentalSpectrum,
    constant: str = "kcal/mol",
) -> np.array:
    spectra_accumulator = es.simulated_vals(boltzmann_weights(x_energies, constant))
    return fitness_tanimoto(spectra_accumulator, es.vals(es.freq_range))


class GeneticProblem(FunctionalProblem):
    def __init__(self, energies: dict, error: float, objs: Callable[[np.ndarray], float]):
        x_l, x_u = energies - error, energies + error
        super().__init__(n_var=len(energies), objs=objs, xl=x_l, xu=x_u, type_var=float)
