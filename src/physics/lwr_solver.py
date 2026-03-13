import numpy as np 

class LWRTrafficFluid:
    """
    Simulates traffic as a compressible fluid across a topological graph.
    Uses the Lighthill-Whitham-Richards (LWR) kinetic wave model.
    """
    def __init__(self, adjacency_matrix: np.ndarray, v_max: float = 15.0, rho_max: float = 1.0):
        self.A = adjacency_matrix
        self.N = self.A.shape[0]

        self.edges = self.A > 0

        self.v_max = v_max
        self.rho_max = rho_max

        self.density = np.random.uniform(0.1, 0.4, self.N)

    def step(self, dt: float = 0.1) -> np.ndarray:
        """
        Executes one finite-volume time step of the fluid simulation.
        Calculates the flow between all connected nodes simultaneously.
        """
        rho_i = self.density[:, np.newaxis]
        rho_j = self.density[np.newaxis, :]

        flux = self.v_max * rho_i * (1.0 - (rho_j / self.rho_max))

        actual_flow = flux * self.edges

        inflow = np.sum(actual_flow, axis=0)
        outflow = np.sum(actual_flow, axis=1)

        d_rho = (inflow - outflow) * dt

        self.density = np.clip(self.density + d_rho, 0.0, self.rho_max)

        return self.density.copy()