from typing import Any, Callable, List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage

from objective.objective import Objective
from spectrum.experimental_spectrum import ExperimentalSpectrum


class ClusteringObjective(Objective):
    def __init__(
        self,
        energies_array: npt.NDArray[np.float64],
        candidates: List[ExperimentalSpectrum],
        error: float,
        energy_unit: str,
        fitness_function: Callable[[npt.NDArray[np.float64]], float],
        reference_candidate: ExperimentalSpectrum,
        cluster_metric: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
        cut_point: float,
        draw_dendrogram: bool,
    ) -> None:
        self._energies_array = energies_array
        self._optimization_candidates = candidates
        self._error = error
        self._eu = energy_unit
        self.fitness = fitness_function

        self.clusters = cut_tree(
            Z=linkage(
                reference_candidate.broadened_vals, metric=cluster_metric, method="complete", optimal_ordering=False
            ),
            height=cut_point,
        ).reshape(-1)

        if draw_dendrogram:
            self.draw_dendrogram(
                list(reference_candidate.broadened),
                linkage(
                    reference_candidate.broadened_vals, metric=cluster_metric, method="complete", optimal_ordering=False
                ),
                cut_point,
            )

        self.chromosome = np.array(
            [np.mean(self._energies_array[self.clusters == cluster]) for cluster in np.sort(np.unique(self.clusters))]
        )

    def draw_dendrogram(self, labels: List[str], linkage: Any, cut_point: float):
        plt.axhline(y=cut_point, c="k", linewidth=0.5)
        dendrogram(linkage, labels=labels, leaf_rotation=90)
        plt.show()

    @property
    def lower(self) -> npt.NDArray[np.float64]:
        return self.chromosome - self._error

    @property
    def upper(self) -> npt.NDArray[np.float64]:
        return self.chromosome + self._error

    @property
    def n_var(self) -> int:
        return self.chromosome.size

    def get_chromosome(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x[self.clusters]

    def process(self, x: npt.NDArray[np.float64]) -> np.float64:
        return -np.prod(
            [self.fitness(self.get_chromosome(x), candidate, self._eu) for candidate in self._optimization_candidates]
        )