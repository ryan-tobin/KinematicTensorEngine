import numpy as np 

class LeakyReLU:
    """
    Solves the Dying ReLU problem by allowing a small, non-zero gradient 
    to flow backward when the network dips into negative values.
    """
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.inputs = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Calculates the gradient of the loss. 
        Never returns exactly 0.0, preventing network brain death.
        """
        d_input = d_out.copy()
        d_input[self.inputs <= 0] *= self.alpha
        return d_input
    
class MSELoss:
    """
    Calculates the penalty when the AI's traffic prediciton diverges from reality.
    """
    def __init__(self):
        self.predictions = None 
        self.targets = None 

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets)**2)
    
    def backward(self) -> np.ndarray:
        """
        The derivative: 2 * (predictions - targets) / N
        """
        n = self.predictions.shape[0]
        return 2.0 * (self.predictions - self.targets) / n 