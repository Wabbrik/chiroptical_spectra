import env  # isort:skip

import sys
from os import getcwd
from os.path import join
from typing import Dict, Union

from pymoo.termination import get_termination

from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.genetic_problem import GeneticProblem, classic_fitness
from objective.classic_objective import ClassicObjective
from objective.clustering_objective import ClusteringObjective
from overlap.metrics import dendrogram_tanimoto
from parameters.input_parameters import InputParameters
from parameters.utils import draw_dendrogram, plot_results, write_dendrogram_data, write_results


def main() -> int:
    ip = InputParameters(path=join(getcwd(), "GA_Analysis_File.json"))
    objectives: Dict[str, Union[ClusteringObjective, ClassicObjective]] = {
        "clustering": ClusteringObjective(
            ip.energies_array(),
            ip.candidates,
            ip.energy_uncertainty,
            ip.eu,
            classic_fitness,
            reference_candidate=ip.reference_candidate,
            cluster_metric=dendrogram_tanimoto,
            cut_point=ip.dendrogram_threshold,
        ),
        "classic": ClassicObjective(
            ip.energies_array(),
            ip.candidates,
            ip.energy_uncertainty,
            ip.eu,
            classic_fitness,
        ),
    }

    if ip.draw_dendrogram:
        dendro = draw_dendrogram(
            list(ip.reference_candidate.broadened), objectives["clustering"].linkage, ip.dendrogram_threshold
        )
        write_dendrogram_data(dendro, join(getcwd(), "dendrogram_ordering.txt"))

    res = GeneticAlgorithm(
        ga_type=ip.genetic_algorithm,
        genetic_problem=GeneticProblem(objectives[ip.objective]),
    ).run(termination=get_termination("n_gen", ip.termination_criterion_ngen))

    print(f"Runtime was: {res.exec_time:.2f} seconds.")

    if not ip.skip_print:
        write_results(
            path=getcwd(),
            fitness=res.F[0],
            key_energies=ip.energies,
            energies=objectives[ip.objective].get_chromosome(res.X),
            constant=ip.eu,
        )
        plot_results(
            path=getcwd(),
            experimental_spectra=ip.candidates,
            energies=objectives[ip.objective].get_chromosome(res.X),
            constant=ip.eu,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\nResulted error: <{type(e).__name__}> ({e})\n")

        if len(sys.argv) > 1 and sys.argv[1] == "trace":
            from traceback import format_exc

            sys.stderr.write(f"\n{format_exc()}\n")

        sys.exit(1)
