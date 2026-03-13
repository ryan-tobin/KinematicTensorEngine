import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class GraphRenderer:
    """
    Translates the mathematical tensor engine back into physical geographic space.
    Renders high-performance thermal heatmaps of the fluid dynamics.
    """
    def __init__(self, parser):
        self.parser = parser
        
        # 1. Map matrix indices back to geographic coordinates
        # parser.node_ids is ordered exactly like the matrix rows/columns
        self.lats = np.array([self.parser.node_coords[nid][0] for nid in self.parser.node_ids])
        self.lons = np.array([self.parser.node_coords[nid][1] for nid in self.parser.node_ids])
        
        # 2. Extract the topology (edges) for the LineCollection
        self.edges = self._build_edge_geometry()

    def _build_edge_geometry(self) -> list:
        """
        Scans the Adjacency Matrix to find connected streets and builds 
        the line segments for lightning-fast rendering.
        """
        print("Compiling spatial geometry for the renderer...")
        segments = []
        # Find all row, col indices where distance > 0 (meaning a street exists)
        rows, cols = np.where(self.parser.adjacency_matrix > 0)
        
        # We only need one direction for drawing the line (row < col avoids duplicates)
        for r, c in zip(rows, cols):
            if r < c:
                # Coordinate format: (longitude, latitude) for standard X, Y plotting
                start_point = (self.lons[r], self.lats[r])
                end_point = (self.lons[c], self.lats[c])
                segments.append([start_point, end_point])
                
        return segments

    def plot_traffic_state(self, density: np.ndarray, title: str = "Macroscopic Traffic Fluid State"):
        """
        Paints the traffic density onto the topological grid.
        Dark purple = empty streets. Yellow = absolute gridlock.
        """
        # Enforce dark mode for high-contrast thermal mapping
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 1. Draw the street network
        # A faint, semi-transparent web representing the Adjacency Matrix
        lc = LineCollection(self.edges, colors='gray', linewidths=0.5, alpha=0.3)
        ax.add_collection(lc)
        
        # 2. Draw the intersections (Nodes)
        # We use the density array to map colors using the 'plasma' colormap
        # We flatten the density array to ensure it maps correctly to the scatter plot
        scatter = ax.scatter(self.lons, self.lats, 
                             c=density.flatten(), cmap='plasma', 
                             s=15, alpha=0.9, zorder=5)
        
        # 3. Format the geographic arena
        ax.set_title(title, fontsize=16, pad=20, color='white')
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        
        # Ensure the aspect ratio matches the physical Earth
        ax.set_aspect('equal')
        
        # Add the thermal scale
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.036, pad=0.04)
        cbar.set_label(r"Fluid Density ($\rho$)", color='white', fontsize=12)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        plt.tight_layout()
        plt.show()