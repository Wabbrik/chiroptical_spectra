from abc import abstractmethod
from typing import List

import numpy as np

from spectrum.experimental_spectrum import ExperimentalSpectrum


class GeneticAlgorithmObjective(object):
    def __init__(self, candidates: List[ExperimentalSpectrum], energy_unit: str) -> None:
        self.optimization_candidates = candidates
        self.energy_unit = energy_unit

    def __call__(self, x):
        return self.process(x)

    @abstractmethod
    def get_chromosome(self) -> np.array:
        raise NotImplementedError()

    @abstractmethod
    def process(self, x):
        raise NotImplementedError()
