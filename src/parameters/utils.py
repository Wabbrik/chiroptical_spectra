import re
from os.path import join
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

from overlap.metrics import tanimoto
from overlap.weights import boltzmann_weights
from spectrum.experimental_spectrum import ExperimentalSpectrum
from spectrum.spectrum_type import title_by_type


def commit_mapping_to_file(fpath: str, mapping: Dict[any, float]):
    with open(fpath, 'w') as f:
        for key, value in mapping.items():
            f.write(f"{key} {value:.4f}\n")


def write_results(
        path: str,
        fitness: float,
        key_energies: List[str],
        energies: List[float],
        constant: float
) -> None:
    for name, values in [("gaEs", energies), ("gaBWs", boltzmann_weights(energies, constant))]:
        try:
            commit_mapping_to_file(file_path := join(path, f"{name}{-fitness:.3f}"), dict(zip(key_energies, values)))
            print(f"{name} written at: {file_path}")
        except Exception as e:
            print(f"Failed to write {name}. Exception: {str(e)}")


def exp_spectrum_plot(path: str, spectrum: ExperimentalSpectrum, energies: List[float], constant: float) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(spectrum.freq(spectrum.freq_range), spectrum.vals(spectrum.freq_range), color='k')
    plt.plot(spectrum.freq(spectrum.freq_range), sim := spectrum.simulated_vals(
        boltzmann_weights(energies, constant)), color='r')
    tsi = tanimoto(sim, spectrum.vals(spectrum.freq_range))
    plt.legend([f"{spectrum.type.name} Experiment", "GA-VCD Boltzmann average"])
    plt.title(f"TSI: {tsi:1.3f}")
    plt.xlabel("Frequency(cm$^{-1}$)")
    plt.ylabel(title_by_type[spectrum.type])
    plt.savefig(figpath := join(path, figname := f"{tsi:1.3f}_{spectrum.type.name}_plot.pdf"))
    print(f"{figname} written at: {figpath}")
    plt.show()


def plot_results(path: str, experimental_spectra: List[ExperimentalSpectrum], energies: List[float], constant: float) -> None:
    for spectrum in experimental_spectra:
        exp_spectrum_plot(path, spectrum, energies, constant)
