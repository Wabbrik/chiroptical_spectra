import sys
from os import getcwd
from os.path import join

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.genetic_problem import GeneticProblem, constants
from objective.concrete.cluster_fitness_objective import ClusterFitnessObjective
from objective.concrete.prod_fitness_objective import ProdFitnessObjective
from parameters.input_parameters import InputParameters
from parameters.utils import plot_results, write_results


def main() -> int:
    ip = InputParameters(path=join(getcwd(), "GA_Analysis_File.json"))
    obj = ClusterFitnessObjective(
        candidates=ip.optimization_candidates(),
        energy_unit=ip.energy_unit,
        cut_point=0.7,
    )
    # obj = ProdFitnessObjective(
    #     ip.optimization_candidates(), ip.energy_unit,
    # )
    genetic_algorithm = GeneticAlgorithm(
        ga_type=ip.genetic_algorithm,
        genetic_problem=GeneticProblem(
            energies=obj.get_chromosome(),
            error=ip.energy_uncertainty,
            objs=obj
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

        plot_results(
            path=getcwd(),
            experimental_spectra=ip.experimental_spectra,
            energies=res.X,
            constant=constants[ip.energy_unit]
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
