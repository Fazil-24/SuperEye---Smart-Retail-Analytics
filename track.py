import csv
import imutils
import cv2
import json
import numpy as np
from scipy.spatial.distance import euclidean
from colors import RGB_COLORS, gradient_color_RGB
from vid_configuration import VIDEO_CONFIG


def visualize_movement_tracks(output_path):
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
    (ret, tracks_frame) = cap.read()
    tracks_frame = imutils.resize(tracks_frame, width=frame_size)

    # Set parameters for movement tracking
    stationary_threshold_seconds = 2
    stationary_threshold_frame = round(
        vid_fps * stationary_threshold_seconds / data_record_frame)
    stationary_distance = frame_size * 0.05

    # Process movement points (filter out stationary points)
    movement_points = []
    for movement in tracks:
        temp_movement_point = [movement[0]]
        stationary = movement[0]
        stationary_time = 0
        for i in movement[1:]:
            if euclidean(stationary, i) < stationary_distance:
                stationary_time += 1
            else:
                temp_movement_point.append(i)
                stationary = i
                stationary_time = 0
        movement_points.append(temp_movement_point)

    # Configure colors for visualization
    color1 = (255, 96, 0)  # Orange
    color2 = (0, 28, 255)  # Blue

    # Draw movement tracks
    line_thickness = 2
    for track in movement_points:
        for i in range(len(track) - 1):
            if len(track) > 1:
                color = gradient_color_RGB(color1, color2, len(track) - 1, i)
                cv2.line(tracks_frame, tuple(track[i]), tuple(
                    track[i+1]), color, line_thickness)

    # Display result
    #cv2.imshow("Movement Tracks", tracks_frame)
    cv2.imwrite("static/track.jpg", tracks_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    visualize_movement_tracks()
