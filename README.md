# Kinematic Tensor Engine

A high-performance, from-scratch **Graph Attention Network (GAT)** designed to predict macroscopic urban traffic fluid dynamics using real-world OpenStreetMap (OSM) topologies.

## Project Navigation
- [Full Research Article](./white-paper.md)
- [MIT License](./LICENSE)
- [Environment Setup](./requirements.txt)

## Overview
The Kinematic Tensor Engine bypasses traditional point-to-point pathfinding to treat urban traffic as a continuous, compressible fluid. By merging the **Lighthill-Whitham-Richards (LWR)** kinematic wave model with a custom sparse-matrix AI core, the engine can predict urban gridlock shockwaves with $O(E)$ computational efficiency.

## Tech Stack
- **AI Core:** Custom Graph Attention Network (GAT) with Sparse Message Passing.
- **Physics:** LWR Fluid Dynamics Simulator (Finite Volume Method).
- **Backend:** FastAPI (Asynchronous Tensor API).
- **Frontend:** Leaflet.js + Leaflet-Geoman (Interactive Thermal Mapping).
- **Data:** OpenStreetMap Overpass API.

## Key Features
- **Zero Frameworks:** Built entirely in NumPy. No PyTorch, no TensorFlow. No black boxes.
- **Sparse Optimization:** Scaled to handle 15,490 nodes and 19,299 edges (Philadelphia Center City) via index-based scatter operations.
- **Dynamic Re-initialization:** Use the on-map drawing tools to select any region on Earth. The system will dynamically fetch OSM data, compile a new adjacency matrix, and re-boot the neural network.
- **Thermal Visualization:** Interactive "Plasma" colormap rendering traffic density in real-time.

## 📖 Deep Dive & Research
For a comprehensive breakdown of the multivariable calculus, fluid dynamics, and neural architecture used in this project, read the full paper:
👉 [Research & Methodology Article](./white-paper.md)

## Getting Started

### 1. Clone & Install
```bash
git clone https://github.com/ryan-tobin/KinematicTensorEngine.git
cd KinematicTensorEngine
pip install -r requirements.txt
```

### 2. Ignite the Tensor API
```bash
# Ensure PYTHONPATH is set to the current directory
PYTHONPATH=. uvicorn src.api:app --reload
```

### 3. Launch the Dashboard

Simply open `dashboard.html` in any modern web browser.

* **Simulate**: Click "Simulate Next Frame" to advance the physics and train the GAT.
* **Change Cities**: Use the rectangle tool on the left to draw a new bounding box anywhere on the map.

## Performance Benchmark (Philadelphia Stress Test)
* **Nodes**: 15,490
* **Topological Edges**: 19,299
* **Training Latency**: ~0.54s per epoch (Local CPU)
* **Mathematical Accuracy**: 0.252 MSE Loss
* **Complexity**: $O(E)$ (Linear with the number of streets)

## License

Distributed under the MIT License. See [License](./LICENSE) for more information.

**Author**: Ryan Tobin
