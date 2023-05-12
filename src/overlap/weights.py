import numpy as np

constants = {
    "kcal/mol": 0.0433640,
    "kj/mol": 0.01036,
    "ev": 1.0,
    "hartree": 27.211396132,
}


def boltzmann_weights(energies: np.array[float], constant: str = "kcal/mol") -> np.ndarray[float]:
    np.exp((energies - min(energies)) * constants[constant] /
           0.02569260860624242419, out=energies)
    rcp_s, min_i = 1 / np.sum(1 / energies), energies.argmin()
    energies, energies[min_i] = rcp_s / energies, rcp_s
    return energies
