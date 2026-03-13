import numpy as np 

class SphericalGeometry:
    """
    Pure mathematical engine for non-Euclidean geospatial calculations.
    Processes massive arrays for lat/lon coordinates at C-level speeds.
    """
    EARTH_RADIUS_METERS = 6371000.0

    @staticmethod
    def haversine_distance(lat1: np.ndarray, lon1: np.ndarray,
                           lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Calculates the Great-Circle distance between two sets of coordinates.
        Fully vectorized to calculate thousands of topological edges simultaneously.

        Args:
            lat1, lon1: Origin coordinates 
            lat2, lon2: Destination coordinates

        Returns:
            np.ndarray: The physical distance in meters
        """
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0)**2 + \
            np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
        
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

        return SphericalGeometry.EARTH_RADIUS_METERS * c