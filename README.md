# Object Tracking with Weighted Graph

## Overview
This project implements **object tracking using YOLOv12** and visualizes movement with a **weighted graph**. The graph dynamically connects detected objects between frames, assigning **edge weights based on Euclidean distance** (pixel movement).

## Features
✅ **Real-time object detection** using YOLOv12.  
✅ **Object tracking** with bounding boxes and movement paths.  
✅ **Weighted graph representation** for object movement and distance calculation between tracked object.  
✅ **Custom adjacency list graph** (no external libraries like `networkx`).  
✅ **Graph visualization with pixel distance annotations**.

## How It Works
1. **Create YOLOv12 model using PyTorch** and processes video frames.
2. **Loads YOLOv12 model** and processes video frames.
3. **Detects objects**, extracts bounding boxes, and calculates center points.
4. **Tracks movement** by linking centers across frames.
5. **Builds a weighted graph**, storing distances between tracked points.
6. **Draws tracking lines**, annotating edges with pixel distances.
7. **Saves the processed video** with bounding boxes and movement paths.

## Graph Representation
- **Nodes** → Object positions.
- **Edges** → Movement between consecutive frames.
- **Weights** → Euclidean distance (movement in pixels).

Example graph representation in Python:
```python
# Adjacency list structure
graph = {
    1: [(2, 5), (3, 8)],  # Node 1 connects to 2 and 3 with weights 5 and 8
    2: [(1, 5), (4, 6)],
    3: [(1, 8)],
    4: [(2, 6)]
}
```

## Output Example
🔹 Bounding boxes drawn on detected objects.  
🔹 Red tracking lines connecting object positions.  
🔹 Pixel movement distance displayed above each tracking line.  

## Future Improvements
🔹 Implement **Dijkstra’s shortest path** for movement analysis.  
🔹 Add **object re-identification (Re-ID)** for lost objects.  
🔹 Extend support for **multiple object tracking (MOT)**.  




https://github.com/user-attachments/assets/c6222f4c-7579-4167-af9a-325d61ade963

