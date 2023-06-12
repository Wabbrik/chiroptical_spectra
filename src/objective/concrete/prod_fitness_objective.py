import numpy as np

from genetic_algorithm.genetic_problem import fitness
from objective.objective import GeneticAlgorithmObjective


class ProdFitnessObjective(GeneticAlgorithmObjective):
    def get_chromosome(self) -> np.array:
        return self.optimization_candidates[0].energies_array()

    def process(self, x):
        return - np.prod([fitness(x, candidate, self.energy_unit) for candidate in self.optimization_candidates])
