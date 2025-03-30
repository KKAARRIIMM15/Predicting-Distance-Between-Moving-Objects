# Object Tracking with Weighted Graph Data Structure

## Overview
 This project shows how to use Pytorch to implement **object tracking using YOLOv12** to visualize movement and calculate distances among all tracked objects with **weighted graph data structure.** The weighted graph dynamically connects detected objects between frames, assigning edge weights based on Euclidean distance (pixel movement).

## Why Use a Weighted Graph Data Structure?
   **weighted graph** is used in this project because:
- It **effectively models object movement**, where nodes represent object positions and edges represent distances among them.
- **Edge weights store movement distance**, enabling advanced tracking insights.
- It allows for potential **graph-based algorithms** like shortest path analysis (Dijkstra’s algorithm) and movement prediction.
- **Scalability:** Can be extended for multiple-object tracking (MOT) with real-time updates.


## Features
✅ **Real-time object detection** using YOLOv12.  
✅ **Object tracking** with bounding boxes and movement paths.  
✅ **Weighted graph representation** for object movement and distance calculation between tracked objects.   
✅ **Graph visualization with pixel distance annotations**.

## How It Works
1. **Create YOLOv12 model using PyTorch**
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
      # Node 1 connects to 2,3 and 4 → with weights 50,80 and 70
    1: [(2, 50), (3, 80), (4, 70), ...etc],  
    2: [(1, 50), (4, 60), ...etc],
    3: [(1, 80), ...etc],
    4: [(2, 60), ...etc]
 ...etc:
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

# Output


https://github.com/user-attachments/assets/dc4fbb12-bd89-4ecd-a031-f18fbfc9a498


https://github.com/user-attachments/assets/eaa6e06b-529e-4f39-82f0-9aecb68fb74d



https://github.com/user-attachments/assets/56799321-93ff-43b0-bce8-fe68ec9591b9






https://github.com/user-attachments/assets/e76a4ae8-527b-44cc-a0d2-3d670d4644c1






https://github.com/user-attachments/assets/cd0977cb-af1b-4572-b4cc-c93556064892



