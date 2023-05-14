import re
from os.path import join
from typing import Dict, List

import numpy as np

from overlap.weights import boltzmann_weights
from spectrum.experimental_spectrum import ExperimentalSpectrum


def get_first_integer(s: str) -> int:
    match = re.match(r'\D*(\d+)', s)
    return int(match.group(1)) if match else 0


def mapping_to_write(experimental_spectrum: ExperimentalSpectrum, values: np.ndarray) -> Dict[int, float]:
    return dict(zip([
        get_first_integer(fname) for fname in list(experimental_spectrum.energies.keys())
    ], values))


def commit_mapping_to_file(fpath: str, mapping: Dict[any, float]):
    with open(fpath, 'w') as f:
        for key, value in mapping.items():
            f.write(f"{key} {value:.4f}\n")


def write_results(
        path: str,
        fitness: float,
        experimental_spectra: List[ExperimentalSpectrum],
        energies: List[float],
        constant: float
) -> None:
    for name, values in [("gaEs", energies), ("gaBWs", boltzmann_weights(energies, constant))]:
        try:
            commit_mapping_to_file(
                file_path := join(path, f"{name}{-fitness:.3f}"), mapping_to_write(experimental_spectra[0], values)
            )
            print(f"{name} written at: {file_path}")
        except Exception as e:
            print(f"Failed to write {name}. Exception: {str(e)}")
