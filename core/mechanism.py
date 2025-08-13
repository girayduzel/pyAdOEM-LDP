#aAdOEM_LD/core/mechanism.py
"Abstract class for LDP mechanisms."

from __future__ import annotations
import abc
import numpy as np
from numpy.typing import NDArray


class Mechanism(abc.ABC):

    def __init__(self, K: int):
        self.K: int = K

    @abc.abstractmethod
    def privatise(self, x: int) -> int: ...
    
    @abc.abstractmethod
    def likelihood_row(self, y: int) -> NDArray[np.float64]: ...