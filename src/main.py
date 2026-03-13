import numpy as np 
import time
from src.geospatial.osm_parser import OSMGraphParser
from src.physics.lwr_solver import LWRTrafficFluid
from src.agent.gcn_policy import TrafficPredictorGCN
from src.visualization.renderer import GraphRenderer

def ignite_tensor_engine(epochs: int = 500):
    # NEW: Start the master execution clock
    start_time = time.time()
    
    print("Initializing Kinematic Graph Tensor Engine...\n")
    
    # ... [Keep your Philadelphia bounding box and initialization code exactly the same] ...
    philly_bbox = (39.945, -75.180, 39.960, -75.140)
    parser = OSMGraphParser(bbox=philly_bbox)
    parser.fetch_urban_grid()
    
    adjacency_matrix = parser.adjacency_matrix
    n_nodes = adjacency_matrix.shape[0]
    
    if n_nodes == 0:
        print("Error: No street nodes found in this bounding box.")
        return

    print("\nBooting the Thermal Rendering Engine...")
    renderer = GraphRenderer(parser)
    
    print("Initializing Macroscopic Fluid Dynamics Simulator...")
    physics_engine = LWRTrafficFluid(adjacency_matrix)
    
    print("Initializing Graph Convolutional Network (GCN)...")
    agent = TrafficPredictorGCN(adjacency_matrix)
    
    # 3. THE TRAINING LOOP
    print("\nBeginning Training Sequence: Predicting Fluid-Dynamic Shockwaves")
    print("-" * 80)
    # NEW: Expanded header to include Elapsed Time
    print(f"{'Epoch':<10} | {'MSE Loss':<20} | {'Status':<15} | {'Elapsed Time'}")
    print("-" * 80)
    
    current_traffic = physics_engine.density.copy()
    
    for epoch in range(epochs):
        future_traffic = physics_engine.step(dt=0.1)
        loss = agent.train_step(current_traffic, future_traffic)
        current_traffic = future_traffic.copy()
        
        # NEW: Calculate how long the engine has been running
        elapsed = time.time() - start_time
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins):02d}:{secs:05.2f}"
        
        # NEW: Print the metrics for EVERY epoch, not just every 100
        status = "CONVERGING" if loss < 0.05 else "LEARNING"
        print(f"{epoch + 1:<10} | {loss:<20.6f} | {status:<15} | {time_str}")

    print("\nEngine Execution Complete. The AI has learned the Philadelphia topology.")
    print("Rendering final traffic fluid state...")
    renderer.plot_traffic_state(current_traffic, title="Philadelphia Gridlock: Final Fluid State")

if __name__ == "__main__":
    ignite_tensor_engine(epochs=500)