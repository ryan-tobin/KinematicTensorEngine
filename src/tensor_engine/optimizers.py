import numpy as np

class AdamOptimizer:
    """
    Adaptive Moment Estimation.
    Dynamically scans layers for any trainable parameters (Weights, Attention Vectors) 
    and applies custom momentum scaling to every single variable.
    """
    def __init__(self, layers: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.layers = layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 

    def step(self):
        self.t += 1
        
        for layer in self.layers:
            # We dynamically check for all possible parameters in our custom layers
            params = [('weights', 'dW'), ('a_src', 'da_src'), ('a_dst', 'da_dst')]
            
            for p_name, dp_name in params:
                if hasattr(layer, p_name) and hasattr(layer, dp_name):
                    param = getattr(layer, p_name)
                    grad = getattr(layer, dp_name)
                    
                    if grad is not None:
                        # Initialize dynamic momentum caches
                        m_name, v_name = f"m_{p_name}", f"v_{p_name}"
                        if not hasattr(layer, m_name):
                            setattr(layer, m_name, np.zeros_like(param))
                            setattr(layer, v_name, np.zeros_like(param))
                            
                        m = getattr(layer, m_name)
                        v = getattr(layer, v_name)
                        
                        # Calculate 1st and 2nd moments
                        m = self.beta1 * m + (1.0 - self.beta1) * grad
                        v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
                        
                        # Save state
                        setattr(layer, m_name, m)
                        setattr(layer, v_name, v)
                        
                        # Bias correction
                        m_hat = m / (1.0 - self.beta1 ** self.t)
                        v_hat = v / (1.0 - self.beta2 ** self.t)
                        
                        # Apply the Adam step to the raw parameter
                        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                        setattr(layer, p_name, param)