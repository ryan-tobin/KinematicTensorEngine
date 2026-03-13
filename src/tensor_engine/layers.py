import numpy as np 

class GraphConvolution:
    """
    A pure-math Graph Convolutional Layer
    Forces the neural network to learn spatial relationships by aggregating
    features from neighboring topological nodes.
    """
    def __init__(self, adjacency_matrix: np.ndarray, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

        A_tilde = adjacency_matrix + np.eye(adjacency_matrix.shape[0])

        D_tilde = np.sum(A_tilde, axis=1)

        D_inv_sqrt = np.power(D_tilde + 1e-8, -0.5)
        D_inv_sqrt = np.diag(D_inv_sqrt)

        self.A_hat = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)

        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)

        self.inputs = None 
        self.dW = None 

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward Pass: H = A_hat * X * W
        """
        self.inputs = x 

        aggregated_features = self.A_hat.dot(x)

        return aggregated_features.dot(self.weights)
    
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward Pass: Routes the error gradients back through the street topology.
        """
        # dW = X^T * (A_hat^T * d_out)
        routed_error = self.A_hat.T.dot(d_out)

        # dX = (A_hat^T * d_out) * W^T
        d_input = routed_error.dot(self.weights.T)

        return d_input