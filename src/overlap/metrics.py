import numpy as np


def tanimoto(f1: np.array, f2: np.array) -> float:
    return (f1_dot_f2 := f1 @ f2) / ((f1 @ f1) + (f2 @ f2) - np.abs(f1_dot_f2))


def fitness_tanimoto(f1: np.array, f2: np.array) -> float:
    """This version only yields values in the interval [0, 2]; where 0 is a complete mirroring."""
    return tanimoto(f1, f2) + 1


def euclidean(f1: np.array, f2: np.array) -> float:
    return np.linalg.norm(f1 - f2)
