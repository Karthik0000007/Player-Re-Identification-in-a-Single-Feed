# 🎯 Real-Time Player Tracking with YOLOv8 and Deep SORT

This project uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection and [Deep SORT](https://github.com/levan92/deep_sort_realtime) for real-time object tracking. It processes a video to detect and track people frame-by-frame, assigning a unique ID to each person.

## 📂 Project Structure

```bash
├── videos/
│ └── 15sec_input_720p.mp4 # Input video file
├── output/
│ └── output_tracked.avi # Output video with tracking results
├── yolov8n.pt # Pre-trained YOLOv8 Nano model (or another .pt model)
├── main.py # Main tracking script (your Python code)
└── README.md # This file
```

## ✅ Features

- Detects **people only** (`class 0` in COCO dataset)
- Uses Deep SORT with MobileNet-based appearance embedding
- Maintains a memory of unique player IDs
- Ignores very small or very large detections (to reduce false positives)
- Displays and saves a video with bounding boxes and tracking IDs

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Karthik0000007/player-tracking-yolov8.git
cd player-tracking-yolov8
```

### 2. Install Dependencies
- Create a virtual environment
```bash
python -m venv venv
source venv\Scripts\activate

pip install -r requirements.txt
```
### 3. Download the YOLOv8 Model
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```
or from 
```bash
https://github.com/ultralytics/ultralytics
```
## Running the Code
```bash
python main.py
```
- The output will be shown in a window (press q to stop early), and the processed video will be saved to:

```bash
output/output_tracked.avi
```

###  Parameters You Can Tweak

- Detection Confidence Threshold: box.conf.item() < 0.5
- Height Range for Filtering: 80 < h < 400
- Player Memory Timeframe: max_frames_to_remember = 300
- Deep SORT Configuration:
- MAX_AGE, N_INIT, MAX_COSINE_DISTANCE, embedder="mobilenet"

#### Notes

- Tracking is based on appearance and motion. Rapid occlusion or overlapping players may lead to ID switches
- MobileNet is used as the feature extractor. You can replace it with other supported backbones
- Make sure to install dependencies using the same Python version (>=3.8 is recommended)

### Sample Output

- Each detected player is shown with a bounding box and a unique ID