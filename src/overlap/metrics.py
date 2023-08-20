import numpy as np
import numpy.typing as npt


def tanimoto(f1: npt.NDArray[np.float64], f2: npt.NDArray[np.float64]) -> float:
    return (f1_dot_f2 := f1 @ f2) / ((f1 @ f1) + (f2 @ f2) - np.abs(f1_dot_f2))


def fitness_tanimoto(f1: npt.NDArray[np.float64], f2: npt.NDArray[np.float64]) -> float:
    """This version only yields values in the interval [0, 2]; where 0 is a complete mirroring."""
    return tanimoto(f1, f2) + 1


def euclidean(f1: npt.NDArray[np.float64], f2: npt.NDArray[np.float64]) -> float:
    return np.linalg.norm(f1 - f2)
