import env  # isort:skip

import sys
from os import getcwd
from os.path import join

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.genetic_problem import GeneticProblem, classic_fitness
from objective.classic_objective import ClassicObjective
from parameters.input_parameters import InputParameters
from parameters.utils import plot_results, write_results


def main() -> int:
    ip = InputParameters(path=join(getcwd(), "GA_Analysis_File.json"))

    genetic_algorithm = GeneticAlgorithm(
        ga_type=ip.genetic_algorithm,
        genetic_problem=GeneticProblem(
            objective=ClassicObjective(
                ip.energies_array(), ip.candidates, ip.energy_uncertainty, ip.eu, classic_fitness
            )
        ),
    )

    res = genetic_algorithm.run()

    print(f"Runtime was: {res.exec_time:.2f} seconds.")

    if not ip.skip_print:
        write_results(path=getcwd(), fitness=res.F[0], key_energies=ip.energies, energies=res.X, constant=ip.eu)
        plot_results(path=getcwd(), experimental_spectra=ip.candidates, energies=res.X, constant=ip.eu)

    return 0


if __name__ == "__main__":
    sys.exit(main())
