from time import time

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.parameters import hierarchical, set_params
from pymoo.core.result import Result
from pymoo.optimize import minimize
from tqdm import tqdm

from genetic_algorithm.genetic_problem import GeneticProblem
from genetic_algorithm.hyperparameter import (hyperoptimizalble,
                                              hyperparameter_optimize)
from genetic_algorithm.seed import SEED

ga_map = {
    "GA":             GA(),
    "BRKGA":          BRKGA(),
    "DE":             DE(),
    "NEDLER_MEAD":    NelderMead(),
    "PSO":            PSO(),
    "PATTERN_SEARCH": PatternSearch(),
    "ES":             ES(),
}


class GeneticAlgorithm:

    def __init__(self, ga_type: str, genetic_problem: GeneticProblem) -> None:
        self.problem = genetic_problem
        self.algorithm_type = ga_map[ga_type]
        self._apply_hyper_params(ga_type)

    def _apply_hyper_params(self, ga_type: str):
        if hyperoptimizalble(self.algorithm_type):
            with tqdm(total=1) as pbar:
                pbar.set_description(
                    f"Applying Hyperparameter optimization to {ga_type}")
                hyper_res = hyperparameter_optimize(
                    self.algorithm_type, self.problem)
                set_params(self.algorithm_type, hierarchical(hyper_res.X))
                pbar.update(1)

    def run(self, **kwargs) -> Result:
        res: Result = minimize(
            self.problem, self.algorithm_type, seed=SEED, verbose=True, **kwargs
        )
        return res
