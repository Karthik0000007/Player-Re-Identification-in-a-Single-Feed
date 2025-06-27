import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

# ===== Configuration =====
INPUT_VIDEO_PATH = "videos\\15sec_input_720p.mp4"
OUTPUT_VIDEO_PATH = "output\output_tracked.avi"
MODEL_PATH = "yolov8n.pt"


MAX_AGE = 120 
N_INIT = 3     
MAX_COSINE_DISTANCE = 0.2  

# ===== Initialize Models =====
model = YOLO(MODEL_PATH)
tracker = DeepSort(
    max_age=MAX_AGE,
    n_init=N_INIT,
    max_cosine_distance=MAX_COSINE_DISTANCE,
    nn_budget=200,
    embedder="mobilenet",
    half=True
)

# ===== Player Memory System =====
class PlayerMemory:
    def __init__(self):
        self.players = {}  # {track_id: last_seen_frame}
        self.current_frame = 0
        self.max_frames_to_remember = 300  # ~10 seconds at 30fps
    
    def update(self, track_id):
        # Ensure track_id is stored as string for consistency
        self.players[str(track_id)] = self.current_frame
    
    def cleanup(self):
        to_delete = [tid for tid, frame in self.players.items() 
                    if self.current_frame - frame > self.max_frames_to_remember]
        for tid in to_delete:
            del self.players[tid]
    
    def increment_frame(self):
        self.current_frame += 1

player_memory = PlayerMemory()

# ===== Main Processing =====
def main():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {INPUT_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            player_memory.increment_frame()
            player_memory.cleanup()

            # Detect players
            results = model(frame, classes=0)  # Only detect people (class 0)
            detections = []
            
            for box in results[0].boxes:
                if box.conf.item() < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                if not (80 < h < 400):
                    continue

                player_crop = frame[y1:y2, x1:x2]
                if player_crop.size == 0:
                    continue

                detections.append(([x1, y1, w, h], box.conf.item(), player_crop))

            
            tracks = tracker.update_tracks(detections, frame=frame)

            
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = str(track.track_id)  # Convert to string for consistency
                player_memory.update(track_id)

                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Generate color based on track_id (convert to int first)
                try:
                    color = (0, 255, 0) if int(track_id) % 2 else (0, 0, 255)
                except ValueError:
                    color = (0, 255, 255)  # Fallback color if track_id isn't numeric

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"ID: {track_id}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1-label_height-5), (x1+label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            out.write(frame)
            cv2.imshow("Player Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Tracked {len(player_memory.players)} unique players")

if __name__ == "__main__":
    main()