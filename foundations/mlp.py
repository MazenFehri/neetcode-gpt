import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        
        for layer in range(len(weights)-1):
            z = np.dot(x , weights[layer])+biases[layer]
            x = np.maximum(0,z)
        y = np.dot(x,weights[-1])+biases[-1]
        return np.round(y,5)