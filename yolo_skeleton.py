import cv2
import time
import sys
import os
import csv
from ultralytics import YOLO

# --- CONFIGURATION ---
WEBCAM_INDEX = 0      
IMGSZ = (352, 640)           
MODEL_PATH = 'yolov8n-pose_openvino_model/' 

# Target file path
LOG_DIR = r"C:\Users\jeffr\Desktop\Prison_Demo\shared_data\logs"
CSV_PATH = os.path.join(LOG_DIR, "num_people.csv")

# Sampling rate: 2 times per second = every 0.5 seconds
SAMPLE_INTERVAL = 0.5 

# --- SETUP ---
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
    except OSError as e:
        print(f"[ERROR] Could not create directory: {e}")
        sys.exit(1)

# Check model
if not os.path.exists(MODEL_PATH):
    print(f"[CRITICAL] Model not found at {MODEL_PATH}")
    sys.exit(1)

print(f"[INFO] Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH, task='pose')
print("[INFO] Model loaded. Running in background...")
print(f"[INFO] Saving to: {CSV_PATH}")

def run_tracker():
    # Setup Camera
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    # Initialize CSV header if file doesn't exist
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: Unix Timestamp, Count
            writer.writerow(["Timestamp", "Person_Count"])

    last_log_time = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            current_time = time.time()
            
            # Check if 0.5s has passed since last log
            if (current_time - last_log_time) >= SAMPLE_INTERVAL:
                
                # Run Inference
                results = model.track(
                    frame, 
                    persist=True, 
                    verbose=False, 
                    imgsz=IMGSZ,
                    conf=0.5
                )

                # Count people
                person_count = 0
                if results[0].boxes.id is not None:
                    person_count = len(results[0].boxes.id)
                
                # Write to CSV
                try:
                    with open(CSV_PATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # current_time is already a float (UNIX timestamp)
                        writer.writerow([current_time, person_count])
                        
                    last_log_time = current_time
                    
                except PermissionError:
                    pass # Fail silently if file is locked by another app

            # Tiny sleep to prevent CPU spiking
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        cap.release()
        print("[INFO] Cleanup complete.")

if __name__ == "__main__":
    run_tracker()