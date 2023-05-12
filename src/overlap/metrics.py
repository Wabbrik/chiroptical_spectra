import numpy as np


def tanimoto(f1: np.array, f2: np.array) -> float:
    return np.dot(f1, f2) / (np.dot(f1, f1) + np.dot(f2, f2) - np.abs(np.dot(f1, f2)))


def euclidean(f1: np.array, f2: np.array) -> float:
    return np.linalg.norm(f1 - f2)
