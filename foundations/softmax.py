import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        max = z.max()
        return np.round(np.exp(z-max) /(np.exp(z-max)).sum() ,4)
