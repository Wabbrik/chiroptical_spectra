import sys
from os import getcwd
from os.path import join

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.genetic_problem import GeneticProblem
from parameters.input_parameters import InputParameters


def main() -> int:
    ip = InputParameters(path=join(getcwd(), "GA_Analysis_File.json"))

    genetic_algorithm = GeneticAlgorithm(
        ga_type=ip.genetic_algorithm,
        genetic_problem=GeneticProblem(
            energies=ip.energies_array(), error=ip.energy_uncertainty, objs=ip.objective_function()
        )
    )
    res = genetic_algorithm.run()

    return 0


if __name__ == '__main__':
    sys.exit(main())
