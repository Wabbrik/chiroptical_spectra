import numpy as np
import numpy.typing as npt

WEIGHT_FOR_CONSTANTS = 1.0 / 0.02569260860624242419

constants = {
    "kcal/mol": 0.0433640 * WEIGHT_FOR_CONSTANTS,
    "kj/mol": 0.01036 * WEIGHT_FOR_CONSTANTS,
    "ev": 1.0 * WEIGHT_FOR_CONSTANTS,
    "hartree": 27.211396132 * WEIGHT_FOR_CONSTANTS,
}


def boltzmann_weights(energies: npt.NDArray[np.float64], constant: str) -> npt.NDArray[np.float64]:
    energies = energies.astype(np.float64)
    energies = np.exp((energies - np.min(energies)) * constants[constant])
    rcp_energies = 1 / energies
    rcp_s = 1 / np.sum(rcp_energies)
    min_i = energies.argmin()
    energies = rcp_s * rcp_energies
    energies[min_i] = rcp_s
    return energies
