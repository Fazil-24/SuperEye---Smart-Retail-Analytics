import csv
import imutils
import cv2
import json
import math
import numpy as np
from scipy.spatial.distance import euclidean
from colors import RGB_COLORS
from vid_configuration import VIDEO_CONFIG


def visualize_heatmap(output_path):
    # Load movement data
    tracks = []
    with open('processed_data/movement_data.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if len(row[3:]) > 4:
                temp = []
                data = row[3:]
                for i in range(0, len(data), 2):
                    temp.append([int(data[i]), int(data[i+1])])
                tracks.append(temp)

    # Load video configuration data
    with open('processed_data/video_data.json', 'r') as file:
        data = json.load(file)
        vid_fps = data["VID_FPS"]
        data_record_frame = data["DATA_RECORD_FRAME"]
        frame_size = data["PROCESSED_FRAME_SIZE"]

    # Get video frame for background
    cap = cv2.VideoCapture(VIDEO_CONFIG["VIDEO_CAP"])
    cap.set(1, 100)  # Set to frame 100
    (ret, heatmap_frame) = cap.read()
    heatmap_frame = imutils.resize(heatmap_frame, width=frame_size)

    # Set parameters for stationary point detection
    stationary_threshold_seconds = 2
    stationary_threshold_frame = round(
        vid_fps * stationary_threshold_seconds / data_record_frame)
    stationary_distance = frame_size * 0.05
    max_stationary_time = 120

    # Set parameters for heatmap visualization
    blob_layer = 50
    max_blob_size = frame_size * 0.1
    layer_size = max_blob_size / blob_layer
    color_start = 210
    color_end = 0
    color_steps = int((color_start - color_end) / blob_layer)
    scale = 1.5

    # Detect stationary points
    stationary_points = []
    for movement in tracks:
        stationary = movement[0]
        stationary_time = 0
        for i in movement[1:]:
            if euclidean(stationary, i) < stationary_distance:
                stationary_time += 1
            else:
                if stationary_time > stationary_threshold_frame:
                    stationary_points.append([stationary, stationary_time])
                stationary = i
                stationary_time = 0

    # Function to draw blob for heatmap
    def draw_blob(frame, coordinates, time):
        if time >= max_stationary_time:
            layer = blob_layer
        else:
            layer = math.ceil(time * scale / layer_size)
        for x in reversed(range(layer)):
            color = color_start - (color_steps * x)
            size = x * layer_size
            cv2.circle(frame, coordinates, int(
                size), (color, color, color), -1)

    # Create heatmap
    heatmap = np.zeros(
        (heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
    for points in stationary_points:
        draw_heatmap = np.zeros(
            (heatmap_frame.shape[0], heatmap_frame.shape[1]), dtype=np.uint8)
        draw_blob(draw_heatmap, tuple(points[0]), points[1])
        heatmap = cv2.add(heatmap, draw_heatmap)

    # Process heatmap
    lo = np.array([color_start])
    hi = np.array([255])
    mask = cv2.inRange(heatmap, lo, hi)
    heatmap[mask > 0] = color_start

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    lo = np.array([128, 0, 0])
    hi = np.array([136, 0, 0])
    mask = cv2.inRange(heatmap, lo, hi)
    heatmap[mask > 0] = (0, 0, 0)

    # Blend heatmap with original frame
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            if (heatmap[row][col] == np.array([0, 0, 0])).all():
                heatmap[row][col] = heatmap_frame[row][col]

    heatmap_frame = cv2.addWeighted(heatmap, 0.75, heatmap_frame, 0.25, 1)

    # Display result
    #cv2.imshow("Stationary Location Heatmap", heatmap_frame)
    cv2.imwrite("static/heatmap.jpg", heatmap_frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    visualize_heatmap()
