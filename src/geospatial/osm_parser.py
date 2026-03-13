import requests 
import numpy as np 
from src.geospatial.geometry import SphericalGeometry

class OSMGraphParser:
    """
    Ingests raw OpenSteetMap data and compiles it into a mathematical Graph.
    Transforms physical streets into an Adjacency Matrix for tensor operations.
    """
    def __init__(self, bbox: tuple):
        self.bbox = bbox 
        self.overpass_url = "http://overpass-api.de/api/interpreter"

        self.node_ids = []
        self.node_coords = {}
        self.adjacency_matrix = None 

    def fetch_urban_grid(self):
        """
        Pings the Overpass API to get every street and intersection in the bounding box.
        """
        print(f"Fetching raw OpenStreetMap data for bounding box: {self.bbox}...")

        # Overpass QL: Get all roads and their connecting nodes
        query = f"""
        [out:json];
        (
            way["highway"]({self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]});
        );
        (._;>;);
        out body;
        """

        response = requests.post(self.overpass_url, data=query)
        if response.status_code != 200:
            raise Exception("Overpass API strictly rejected the request. Check coordinates")
        
        data = response.json()
        self._compile_graph(data['elements'])

    def _compile_graph(self, elements: list):
        """
        Translates the raw JSON into pure NumPy arrays.
        """
        print("Compiling geospatial data into topoogical matrix...")

        ways = []
        for el in elements:
            if el['type'] == 'node':
                self.node_coords[el['id']] = (el['lat'], el['lon'])
                self.node_ids.append(el['id'])
            elif el['type'] == 'way':
                if 'nodes' in el:
                    ways.append(el['nodes'])
        
        n_nodes = len(self.node_ids)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

        id_to_index = {node_id: idx for idx, node_id in enumerate(self.node_ids)}

        edges_built = 0
        for way in ways:
            for i in range(len(way) - 1):
                node_a, node_b = way[i], way[i+1]

                if node_a in id_to_index and node_b in id_to_index:
                    idx_a = id_to_index[node_a]
                    idx_b = id_to_index[node_b]

                    lat_a, lon_a = self.node_coords[node_a]
                    lat_b, lon_b = self.node_coords[node_b]

                    dist = SphericalGeometry.haversine_distance(
                        np.array(lat_a), np.array(lon_a),
                        np.array(lat_b), np.array(lon_b)
                    )

                    self.adjacency_matrix[idx_a, idx_b] = dist 
                    self.adjacency_matrix[idx_b, idx_a] = dist 
                    edges_built += 1
        
        print(f"Graph Compiled: {n_nodes} nodes, {edges_built} topological edges.")
        print(f"Adjacency Matrix Shape: {self.adjacency_matrix.shape}")
        