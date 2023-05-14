import sys
from os import getcwd
from os.path import join

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.genetic_problem import GeneticProblem, constants
from parameters.input_parameters import InputParameters
from parameters.utils import write_results


def main() -> int:
    ip = InputParameters(path=join(getcwd(), "GA_Analysis_File.json"))

    genetic_algorithm = GeneticAlgorithm(
        ga_type=ip.genetic_algorithm,
        genetic_problem=GeneticProblem(
            energies=ip.energies_array(), error=ip.energy_uncertainty, objs=ip.objective_function()
        )
    )

    res = genetic_algorithm.run()

    print(f"Runtime was: {res.exec_time:.2f} seconds.")

    if not ip.skip_print:
        write_results(
            path=getcwd(),
            fitness=res.F[0],
            experimental_spectra=ip.experimental_spectra,
            energies=res.X,
            constant=constants[ip.energy_unit]
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
