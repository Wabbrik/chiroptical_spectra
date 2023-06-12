from collections import defaultdict
from itertools import chain
from typing import Dict, List

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage

from genetic_algorithm.genetic_problem import fitness
from objective.objective import GeneticAlgorithmObjective
from overlap.metrics import tanimoto
from spectrum.experimental_spectrum import ExperimentalSpectrum
from spectrum.spectrum import Spectrum


class ClusterFitnessObjective(GeneticAlgorithmObjective):
    def __init__(self, cut_point: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clusters = self.__get_clusters(cut_point)

    def _average_values(self, clusters: Dict[int, List[str]]) -> np.array:
        clustering_candidate = self.optimization_candidates[0]
        return np.array([np.mean([clustering_candidate.energies[c] for c in cluster_values])
                         for cluster_values in clusters.values()])

    def average_to_energies(self, average_energies: np.array, clusters: Dict[int, List[str]]) -> np.array:
        energies = []
        for i, vals in clusters.items():
            energies.extend([average_energies[i] for _ in vals])
        return np.array(energies)

    def get_energies(self) -> np.array:
        return list(chain.from_iterable(list(self.clusters.values())))

    def get_chromosome(self) -> np.array:
        return self._average_values(self.clusters)

    def __get_clusters(self, cut_point: float) -> Dict[int, List[str]]:
        clustering_candidate: ExperimentalSpectrum = self.optimization_candidates[0]

        intensities = [spec.vals(clustering_candidate.freq_range)
                       for spec in list(clustering_candidate.broadened.values())]

        cutree = cut_tree(Z=linkage(intensities,
                                    metric=tanimoto,
                                    method="complete",
                                    optimal_ordering=False),
                          height=cut_point).reshape(-1)

        cluster_dict = defaultdict(list)
        for cluster, value in zip(cutree, list(clustering_candidate.broadened.keys())):
            cluster_dict[cluster].append(value)

        return cluster_dict

    def process(self, x):
        return - np.prod(
            [fitness(self.average_to_energies(x, self.clusters), candidate, self.energy_unit)
                for candidate in self.optimization_candidates]
        )
