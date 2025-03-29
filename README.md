# Object Tracking with Weighted Graph

## Overview
This project implements **object tracking using YOLOv12** and visualizes movement with a **weighted graph**. The graph dynamically connects detected objects between frames, assigning **edge weights based on Euclidean distance** (pixel movement).

## Features
✅ **Real-time object detection** using YOLOv12.  
✅ **Object tracking** with bounding boxes and movement paths.  
✅ **Weighted graph representation** for object movement.  
✅ **Custom adjacency list graph** (no external libraries like `networkx`).  
✅ **Graph visualization with pixel distance annotations**.

## Installation
### Prerequisites
Ensure you have Python installed and set up a virtual environment.

### Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

## Usage
### Run the Tracking Script
```bash
python object_tracking.py
```

### Modify Paths Before Running
Edit the script to point to your video and model files:
```python
video_path = "D:\\T8\\pythn\\dataset\\Vehicle Dataset Sample 3.mp4"
yolo_model_path = "D:\\T8\\pythn\\dataset\\yolo12n.pt"
```

## How It Works
1. **Loads YOLO model** and processes video frames.
2. **Detects objects**, extracts bounding boxes, and calculates center points.
3. **Tracks movement** by linking centers across frames.
4. **Builds a weighted graph**, storing distances between tracked points.
5. **Draws tracking lines**, annotating edges with pixel distances.
6. **Saves the processed video** with bounding boxes and movement paths.

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
