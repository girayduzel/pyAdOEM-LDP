#AdOEM_LDP/core/estimator.py
"""
Abstract class for online and offline estimators.
"""

from __future__ import annotations
import abc
from typing import Protocol
from numpy.typing import NDArray
import numpy as np


class OnlineEstimator(Protocol):

    @abc.abstractmethod
    def update(self, y: int, g_row: NDArray[np.float64]) -> None: ...

    @abc.abstractmethod
    def current(self) -> NDArray[np.float64]: ...


class OfflineEstimator(Protocol):

    @abc.abstractmethod
    def fit(
        self, ys: list[int], g_rows: list[NDArray[np.float64]]
    ) -> NDArray[np.float64]: ...
