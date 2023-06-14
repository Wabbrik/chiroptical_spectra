from typing import Union

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.algorithms.base.local import LocalSearch
from pymoo.algorithms.hyperparameters import (HyperparameterProblem,
                                              SingleObjectiveSingleRun)
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.result import Result
from pymoo.optimize import minimize

from genetic_algorithm.genetic_problem import GeneticProblem
from genetic_algorithm.seed import SEED


def hyperoptimizable(algorithm_type: Union[GeneticAlgorithm, LocalSearch, Algorithm]) -> bool:
    return not any(isinstance(algorithm_type, cls) for cls in (PatternSearch, PSO, NelderMead, ES))


def hyperparameter_optimize(algorithm_type: Union[GeneticAlgorithm, LocalSearch, Algorithm],
                            problem: GeneticProblem) -> Result:
    return minimize(
        HyperparameterProblem(
            algorithm_type,
            SingleObjectiveSingleRun(
                problem,
                termination=("n_evals", 500),
                seed=SEED
            )
        ),
        Optuna(),
        termination=('n_evals', 50),
        seed=1,
        verbose=False
    )
