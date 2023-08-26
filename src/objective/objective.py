from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Objective(ABC):
    def __call__(self, x: npt.NDArray[np.float64]) -> np.float64:
        return self.process(x)

    @abstractmethod
    def process(self, x: npt.NDArray[np.float64]) -> np.float64:
        raise NotImplementedError()

    @property
    @abstractmethod
    def lower(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def upper(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError()

    @abstractmethod
    def get_chromosome(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_var(self) -> int:
        raise NotImplementedError()
