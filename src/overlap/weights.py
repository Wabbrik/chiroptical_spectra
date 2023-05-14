import numpy as np


def boltzmann_weights(energies: np.ndarray, constant: float) -> np.ndarray:
    energies = energies.astype(np.float32)
    energies = np.exp(
        (energies - np.min(energies)) * constant / 0.02569260860624242419
    )
    rcp_s, min_i = 1 / np.sum(1 / energies), energies.argmin()
    energies, energies[min_i] = rcp_s / energies, rcp_s
    return energies
