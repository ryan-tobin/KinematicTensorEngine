import numpy as np

class GraphAttentionLayer:
    """
    Sparse Graph Attention Mechanism (O(E) Complexity).
    Bypasses dense matrix multiplication by passing messages strictly 
    along physical street edges using NumPy scatter operations.
    """
    def __init__(self, adjacency_matrix: np.ndarray, input_dim: int, output_dim: int, leaky_alpha: float = 0.2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leaky_alpha = leaky_alpha
        self.N = adjacency_matrix.shape[0]
        
        # 1. SPARSE EDGE EXTRACTION
        # Find exactly where the streets exist, ignoring the 99% empty space
        rows, cols = np.where(adjacency_matrix > 0)
        
        # Add self-loops so intersections remember their own traffic state
        self.src = np.concatenate([rows, np.arange(self.N)])
        self.dst = np.concatenate([cols, np.arange(self.N)])
        
        # 2. Initialize Neural Parameters
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.a_src = np.random.randn(output_dim, 1) * np.sqrt(2.0 / output_dim)
        self.a_dst = np.random.randn(output_dim, 1) * np.sqrt(2.0 / output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inputs = x
        
        # Step 1: Linear Transformation
        self.Z = x.dot(self.weights)
        
        # Step 2: Calculate attention scores globally for all nodes
        score_src = self.Z.dot(self.a_src).flatten()
        score_dst = self.Z.dot(self.a_dst).flatten()
        
        # Step 3: SPARSE EDGE COMBINATION
        # We ONLY add scores together if a physical street connects them
        self.E_raw = score_src[self.src] + score_dst[self.dst]
        
        # Step 4: LeakyReLU
        E_leaky = np.where(self.E_raw > 0, self.E_raw, self.leaky_alpha * self.E_raw)
        
        # Step 5: SPARSE SOFTMAX
        # Find the max score per destination node for numeric stability
        e_max = np.zeros(self.N)
        np.maximum.at(e_max, self.dst, E_leaky)
        
        # Exponentiate and calculate the denominator using scatter-add
        e_exp = np.exp(E_leaky - e_max[self.dst])
        e_sum = np.zeros(self.N)
        np.add.at(e_sum, self.dst, e_exp)
        
        # The final attention scores strictly for the valid edges
        self.alpha = e_exp / (e_sum[self.dst] + 1e-8)
        
        # Step 6: SPARSE AGGREGATION (Message Passing)
        # Multiply neighboring traffic by the attention score, and route it to the destination
        out = np.zeros_like(self.Z)
        weighted_Z = self.alpha[:, np.newaxis] * self.Z[self.src]
        np.add.at(out, self.dst, weighted_Z)
        
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Sparse Multivariable Chain Rule using unbuffered scatter-add derivatives.
        """
        dZ = np.zeros_like(self.Z)
        
        # 1. Gradient of Z from the final message passing aggregation
        d_weighted_Z = d_out[self.dst]
        d_alpha = np.sum(d_weighted_Z * self.Z[self.src], axis=1)
        np.add.at(dZ, self.src, d_weighted_Z * self.alpha[:, np.newaxis])
        
        # 2. Sparse Softmax Jacobian
        alpha_d_alpha = self.alpha * d_alpha
        sum_alpha_d_alpha = np.zeros(self.N)
        np.add.at(sum_alpha_d_alpha, self.dst, alpha_d_alpha)
        dE_masked = self.alpha * (d_alpha - sum_alpha_d_alpha[self.dst])
        
        # 3. Route error through LeakyReLU
        dE_raw = np.where(self.E_raw > 0, dE_masked, self.leaky_alpha * dE_masked)
        
        # 4. Route error back to the Source and Destination nodes independently
        d_score_src = np.zeros(self.N)
        np.add.at(d_score_src, self.src, dE_raw)
        d_score_dst = np.zeros(self.N)
        np.add.at(d_score_dst, self.dst, dE_raw)
        
        # 5. Gradients for the Attention Vectors
        self.da_src = self.Z.T.dot(d_score_src[:, np.newaxis])
        self.da_dst = self.Z.T.dot(d_score_dst[:, np.newaxis])
        
        # 6. Add the attention routing error back into the primary Z gradient
        dZ += d_score_src[:, np.newaxis] * self.a_src.T
        dZ += d_score_dst[:, np.newaxis] * self.a_dst.T
        
        # 7. Gradients for the primary Weights and Input
        self.dW = self.inputs.T.dot(dZ)
        d_input = dZ.dot(self.weights.T)
        
        return d_input