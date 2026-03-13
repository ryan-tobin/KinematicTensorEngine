import time 
import numpy as np 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.geospatial.osm_parser import OSMGraphParser
from src.physics.lwr_solver import LWRTrafficFluid
from src.agent.gcn_policy import TrafficPredictorGCN

app = FastAPI(title="Kinematic Graph Tensor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

engine_state = {}

@app.on_event("startup")
async def startup_event():
    print("Booting Kinematic Tensor API...")

    philly_bbox = (39.945, -75.180, 39.960, -75.140)
    parser = OSMGraphParser(bbox=philly_bbox)
    parser.fetch_urban_grid()

    physics = LWRTrafficFluid(parser.adjacency_matrix)
    agent = TrafficPredictorGCN(parser.adjacency_matrix)

    print("Pre-warming the Graph Attention Network...")
    current_traffic = physics.density.copy()
    for _ in range(50):
        future_traffic = physics.step(dt=0.1)
        agent.train_step(current_traffic, future_traffic)
        current_traffic = future_traffic.copy()

    print("API is ready for web requests")

    engine_state['parser'] = parser 
    engine_state['physics'] = physics
    engine_state['agent'] = agent 
    engine_state['current_traffic'] = current_traffic

@app.get("/simulate")
async def simulate_traffic_step():
    """
    Executes one live frame of the simulation and returns the data as GeoJSON
    """
    parser = engine_state['parser']
    physics = engine_state['physics']
    agent = engine_state['agent']
    current_traffic = engine_state['current_traffic']

    future_traffic = physics.step(dt=0.1)

    loss = agent.train_step(current_traffic, future_traffic)
    engine_state['current_traffic'] = future_traffic.copy()

    prediction = agent.forward(current_traffic.reshape(-1, 1)).flatten()

    features = []
    for i, node_id in enumerate(parser.node_ids):
        lat, lon = parser.node_coords[node_id]

        density = float(np.clip(prediction[i], 0.0, 1.0))

        features.append({
            "type": "Feature",
            "geometry": {
                "type" : "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "density": density
            }
        })

    return {
        "status": "success",
        "loss": float(loss),
        "geojson": {
            "type": "FeatureCollection",
            "features": features
        }
    }

class BBoxRequest(BaseModel):
    bbox: list[float]

@app.post("/reinit")
async def reinitialize_engine(data: BBoxRequest):
    print(f"Reinitializing engine for new area: {data.bbox}")
    
    # 1. Parse the new Grid
    new_parser = OSMGraphParser(bbox=tuple(data.bbox))
    new_parser.fetch_urban_grid()
    
    adj = new_parser.adjacency_matrix
    if adj.shape[0] == 0:
        return {"status": "error", "message": "No roads found in this area."}

    # 2. Reset the Physics and AI with the new Adjacency Matrix
    new_physics = LWRTrafficFluid(adj)
    new_agent = TrafficPredictorGCN(adj)
    
    # 3. Update global state
    engine_state['parser'] = new_parser
    engine_state['physics'] = new_physics
    engine_state['agent'] = new_agent
    engine_state['current_traffic'] = new_physics.density.copy()
    
    return {
        "status": "success", 
        "nodes": adj.shape[0], 
        "edges": len(new_parser.node_ids)
    }