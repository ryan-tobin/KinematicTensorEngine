import numpy as np
# We drop the old GraphConvolution and bring in the new GraphAttentionLayer
from src.tensor_engine.attention import GraphAttentionLayer
from src.tensor_engine.activations import LeakyReLU, MSELoss
from src.tensor_engine.optimizers import AdamOptimizer

class TrafficPredictorGCN:
    """
    The central AI Agent. 
    Now powered by a Graph Attention Network (GAT) to dynamically slice 
    through the over-smoothing problem.
    """
    def __init__(self, adjacency_matrix: np.ndarray, node_features: int = 1, hidden_dim: int = 16):
        
        # 1. Assemble the Neural Network Pipeline using Attention
        self.gat1 = GraphAttentionLayer(adjacency_matrix, input_dim=node_features, output_dim=hidden_dim)
        self.relu1 = LeakyReLU()
        self.gat2 = GraphAttentionLayer(adjacency_matrix, input_dim=hidden_dim, output_dim=1)
        
        self.layers = [self.gat1, self.relu1, self.gat2]
        
        # 2. Initialize the Math Engines
        self.loss_fn = MSELoss()
        
        # Throttle the learning rate back slightly to let the Attention Vectors stabilize
        self.optimizer = AdamOptimizer(self.layers, lr=0.001)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_gradient: np.ndarray):
        d_out = loss_gradient
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def train_step(self, current_traffic: np.ndarray, future_traffic: np.ndarray) -> float:
        x = current_traffic.reshape(-1, 1)
        y_true = future_traffic.reshape(-1, 1)
        
        y_pred = self.forward(x)
        loss = self.loss_fn.forward(y_pred, y_true)
        loss_grad = self.loss_fn.backward()
        
        self.backward(loss_grad)
        self.optimizer.step()
        
        return loss