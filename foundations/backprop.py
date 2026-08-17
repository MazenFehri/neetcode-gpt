import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        
        z = np.dot(x,w) +b
        y_pred = 1/(1+np.exp(-z))

        L = 0.5 * np.square(y_pred-y_true) 
        dw = (y_pred-y_true) * y_pred*(1-y_pred) * x
        
        db = (y_pred-y_true) * y_pred*(1-y_pred)

        return (np.round(dw,5), np.round(db,5))
        
