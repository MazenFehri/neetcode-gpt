import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        if activation == "sigmoid":
            return np.round(1 / (1 + np.exp(-(np.sum(np.matmul(x,w))+b))),5)
        return np.round(np.maximum(0,np.sum(np.matmul(x,w))+b),5)

