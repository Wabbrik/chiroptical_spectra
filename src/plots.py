from collections import defaultdict
import sys
from os import getcwd
from os.path import join
import matplotlib.pyplot as plt

from overlap.metrics import tanimoto
from parameters.input_parameters import InputParameters
from parameters.utils import get_first_integer, plot_results, write_results

def main() -> int:
    ip = InputParameters(path=join(getcwd(), "GA_Analysis_File.json"))
    
    candidates = [spectrum for spectrum in ip.experimental_spectra if spectrum.is_opt_candidate]
    correlations = dict()
    for candidate in candidates:
        for key, candidate_broadened in candidate.broadened.items():
            correlation = (tanimoto(candidate.vals(candidate.freq_range), candidate_broadened.vals(candidate.freq_range)) + 1) / 2
            correlations[get_first_integer(key)] = correlation
        correlation_sum = sum(correlations.values())
        correlation_average = correlation_sum / len(correlations)
        for key in correlations.keys():
            correlations[key] = (correlations[key] - correlation_average) # / correlation_sum

    # for correlation in correlations.values():
    #     plt.bar(list(range(len(correlation))), correlation)
    #     plt.show()        
    def load_file(filename: str):
        weights_per_conformer = {}
        with open(filename, "r") as file:
            for line in file.readlines():
                k, value = line.split()
                weights_per_conformer[int(k)] = float(value)

        return weights_per_conformer
        # return list(dict(sorted(weights_per_conformer.items())).values())
    
    weights_list = []
    weights_list.append(load_file("gaBWs0.746"))
    weights_list.append(load_file("gaBWs0.671"))
    weights_list.append(correlations)

    for weights in weights_list:
        plt.bar(list(weights.keys()), list(weights.values()), alpha=0.5)

    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
