import cv2
import numpy as np
from ultralytics import YOLO
from sort.tracker import SortTracker

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure best.pt is in the same directory

# Initialize SORT tracker
tracker = SortTracker(max_age=5, min_hits=3, iou_threshold=0.3)

# Video path
VIDEO_PATH = "basket_detection.mp4"


def detect_baskets(frame):
    """
    Process a single frame for basket detection and tracking.
    """
    global tracker
    
    results = model(frame)  # Run YOLOv8 inference
    detection_list = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            
            if conf > 0.5:
                detection_list.append([x1, y1, x2, y2, conf, cls])
    
    detection_array = np.array(detection_list, dtype=np.float32) if detection_list else np.empty((0, 6), dtype=np.float32)
    tracked_objects = tracker.update(detection_array, None)
    
    unique_basket_ids = set()
    for obj in tracked_objects:
        if len(obj) >= 5:
            x1, y1, x2, y2, obj_id = map(int, obj[:5])
            cls_id = int(obj[5]) if len(obj) > 5 else 0
            conf = float(obj[6]) if len(obj) > 6 else 0.0
            
            unique_basket_ids.add(obj_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {obj_id} Class: {cls_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Baskets count: {len(unique_basket_ids)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame


def main():
    """
    Main function to process the video and display detections.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {VIDEO_PATH}")
    
    frame_count = 0
    unique_basket_ids = set()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame = detect_baskets(frame)
            
            cv2.imshow("Basket Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"\nTotal Unique Baskets Detected: {len(unique_basket_ids)}")
        print(f"Processed {frame_count} frames")
        cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
