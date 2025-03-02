import time
import datetime
import numpy as np
import imutils
import cv2
import os
import csv
import json
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from tracking_helper import detect_human
from vid_configuration import YOLO_CONFIG, VIDEO_CONFIG, SHOW_PROCESSING_OUTPUT, DATA_RECORD_RATE, FRAME_SIZE, TRACK_MAX_AGE, ABNORMAL_ENERGY, ABNORMAL_THRESH, ABNORMAL_MIN_PEOPLE

# Check frame size validity
if FRAME_SIZE > 1920:
    print("Frame size is too large!")
    quit()
elif FRAME_SIZE < 480:
    print("Frame size is too small! You won't see anything")
    quit()

# Load YOLOv3-tiny weights and config
WEIGHTS_PATH = YOLO_CONFIG["WEIGHTS_PATH"]
CONFIG_PATH = YOLO_CONFIG["CONFIG_PATH"]
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]

# Tracker setup
max_cosine_distance = 0.7
nn_budget = None
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age=TRACK_MAX_AGE)

# Ensure data directories exist
os.makedirs('processed_data', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Open files for writing data
movement_data_file = open('processed_data/movement_data.csv', 'a', newline='')
crowd_data_file = open('static/crowd_data.csv', 'a', newline='')

movement_data_writer = csv.writer(movement_data_file)
crowd_data_writer = csv.writer(crowd_data_file)

# Write headers if files are empty
if os.path.getsize('processed_data/movement_data.csv') == 0:
    movement_data_writer.writerow(['Track ID', 'Entry time', 'Exit Time', 'Movement Tracks'])
if os.path.getsize('static/crowd_data.csv') == 0:
    crowd_data_writer.writerow(['Time', 'Human Count', 'Social Distance Violate', 'Restricted Entry', 'Abnormal Activity'])

# Function to process a single frame
def process_frame(frame):
    frame = imutils.resize(frame, width=FRAME_SIZE)
    humans_detected, expired = detect_human(net, ln, frame, encoder, tracker, datetime.datetime.now())
    for track in expired:
        _record_movement_data(track)
    
    # Detect abnormal activity
    abnormal_individuals = []
    for track in humans_detected:
        if len(track.positions) > 1:
            ke = np.linalg.norm(np.array(track.positions[-1]) - np.array(track.positions[-2]))
            if ke > ABNORMAL_ENERGY:
                abnormal_individuals.append(track.track_id)
    abnormal_activity = len(abnormal_individuals) > (len(humans_detected) * ABNORMAL_THRESH)
    
    # Draw bounding boxes
    for track in humans_detected:
        x, y, w, h = map(int, track.to_tlbr())
        color = (0, 255, 0) if track.track_id not in abnormal_individuals else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (w, h), color, 2)
        cv2.putText(frame, f"ID: {track.track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.putText(frame, f"Crowd Count: {len(humans_detected)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if abnormal_activity:
        cv2.putText(frame, "ABNORMAL ACTIVITY DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Record crowd data
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crowd_data_writer.writerow([timestamp, len(humans_detected), 0, 0, int(abnormal_activity)])
    
    return frame, len(humans_detected)

# Function to record movement data
def _record_movement_data(movement):
    with open('processed_data/movement_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        positions = movement.positions if isinstance(movement.positions, list) else list(movement.positions)
        writer.writerow([movement.track_id, movement.entry, movement.exit, len(positions)])

# Function to generate video frames
def generate_crowd_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            print("End of video or empty frame detected.")
            break  # Stop if no more frames
        
        # Ensure the frame is valid
        if frame is None or frame.size == 0:
            print("Skipping empty frame.")
            continue
        
        # Process frame (detection + bounding boxes)
        frame, human_count = process_frame(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if not ret:
            print("Failed to encode frame.")
            continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    movement_data_file.close()
    crowd_data_file.close()
