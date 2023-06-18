from collections import defaultdict
import sys
from os import getcwd
from os.path import join
import matplotlib.pyplot as plt

from overlap.metrics import tanimoto
from parameters.input_parameters import InputParameters
from parameters.utils import plot_results, write_results

def main() -> int:
    ip = InputParameters(path=join(getcwd(), "GA_Analysis_File.json"))
    
    # candidates = [spectrum for spectrum in ip.experimental_spectra if spectrum.is_opt_candidate]
    # correlations = defaultdict(list)
    # for i, candidate in enumerate(candidates):
    #     for candidate_broadened_vals in candidate.broadened_vals:
    #         correlations[i].append(tanimoto(candidate.vals(candidate.freq_range), candidate_broadened_vals))

    # for correlation in correlations.values():
    #     plt.bar(list(range(len(correlation))), correlation)
    #     plt.show()        

    weights_per_conformer = {}
    with open("gaBWs0.786", "r") as file:
        for line in file.readlines():
            k, value = line.split()
            weights_per_conformer[int(k)] = float(value)

    weights = list(dict(sorted(weights_per_conformer.items())).values())
    
    plt.bar(list(range(len(weights))), weights)
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
