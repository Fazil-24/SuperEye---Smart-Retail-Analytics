from flask import Flask, render_template, Response, jsonify,  send_file
import cv2
#from shoplifting_detections import classify_and_detect  # Shoplifting detection function
#from cart_detection import detect_baskets
from crowd import generate_crowd_frames
from heat import visualize_heatmap
from track import visualize_movement_tracks
import os
import csv
import markdown
from GENAI_analysis import generate
import time

app = Flask(__name__)

# Video paths
shoplifting_video_path = '6.mp4'
basket_video_path = 'basket_detection.mp4'

# Video capture objects (initialized when detection starts)
shoplifting_cap = None
basket_cap = None

# Running flags
shoplifting_running = False
basket_running = False

"""
# Generate frames for shoplifting detection
def generate_shoplifting_frames():
    global shoplifting_cap, shoplifting_running
    if not shoplifting_running:
        return

    shoplifting_cap = cv2.VideoCapture(shoplifting_video_path)

    while shoplifting_running:
        success, frame = shoplifting_cap.read()
        if not success:
            break
        frame = classify_and_detect(frame)  # Process frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    shoplifting_cap.release()


# Generate frames for basket detection
def generate_basket_frames():
    global basket_cap, basket_running
    if not basket_running:
        return

    basket_cap = cv2.VideoCapture(basket_video_path)

    while basket_running:
        success, frame = basket_cap.read()
        if not success:
            break
        frame = detect_baskets(frame)  # Process frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    basket_cap.release()
"""

# Home page
@app.route('/')
def crowd():
    return render_template('index.html')

@app.route('/heat')
def heat():
    return render_template('heat.html')

@app.route('/track')
def track():
    return render_template('track.html')

@app.route('/dash')
def dash():
    return render_template('dash.html')

@app.route('/shoplift')
def index():
    return render_template('shoplift.html')

@app.route('/cart')
def cart():
    return render_template('cart.html')



"""
# Video feed for shoplifting detection
@app.route('/video_feed_shoplifting')
def video_feed_shoplifting():
    if not shoplifting_running:
        return jsonify({"error": "Shoplifting detection not started"}), 400
    return Response(generate_shoplifting_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Video feed for basket detection
@app.route('/video_feed_basket')
def video_feed_basket():
    if not basket_running:
        return jsonify({"error": "Basket detection not started"}), 400
    return Response(generate_basket_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Start shoplifting detection
@app.route('/run_shoplifting_detection', methods=['POST'])
def run_shoplifting_detection():
    global shoplifting_running
    shoplifting_running = True
    return jsonify({"message": "Shoplifting detection started"}), 200


# Start basket detection
@app.route('/run_basket_detection', methods=['POST'])
def run_basket_detection():
    global basket_running
    basket_running = True
    return jsonify({"message": "Basket detection started"}), 200
"""



@app.route("/video_feed_crowd")
def video_feed_crowd():
    if not crowd_running:
        return jsonify({"error": "Crowd counting not started"}), 400
    return Response(generate_crowd_frames(basket_video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/run_crowd_count", methods=["POST"])
def run_crowd_count():
    global crowd_running
    crowd_running = True
    return jsonify({"message": "Crowd counting started"}), 200

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    """Runs the heatmap script and saves it as an image"""
    heatmap_path = "static/heatmap.jpg"

    # Generate heatmap
    visualize_heatmap(output_path=heatmap_path)

    if os.path.exists(heatmap_path):
        return jsonify({"message": "Heatmap generated", "image_url": f"/static/heatmap.jpg"}), 200
    else:
        return jsonify({"error": "Failed to generate heatmap"}), 500

@app.route('/get_heatmap')
def get_heatmap():
    """Serves the heatmap image"""
    heatmap_path = "static/heatmap.jpg"
    if os.path.exists(heatmap_path):
        return send_file(heatmap_path, mimetype='image/jpeg')
    return jsonify({"error": "Heatmap not available"}), 404

@app.route('/generate_track', methods=['POST'])
def generate_track():
    """Runs the track script and saves it as an image"""
    track_path = "static/track.jpg"

    # Generate track
    visualize_movement_tracks(output_path=track_path)

    if os.path.exists(track_path):
        return jsonify({"message": "track generated", "image_url": f"/static/track.jpg"}), 200
    else:
        return jsonify({"error": "Failed to generate track"}), 500

@app.route('/get_track')
def get_track():
    """Serves the track image"""
    track_path = "static/track.jpg"
    if os.path.exists(track_path):
        return send_file(track_path, mimetype='image/jpeg')
    return jsonify({"error": "track not available"}), 404


# @app.route('/get_kpi_data')
# def get_kpi_data():
#     """Reads latest KPI values from CSV file"""
#     try:
#         with open("static/kpi_data.csv", "r") as file:
#             reader = csv.reader(file)
#             next(reader)  # Skip header
#             last_row = list(reader)[-1]  # Get latest row
#             data = {
#                 "crowd_count": int(last_row[1]),
#                 "basket_count": int(last_row[2]),
#                 "shoplifting_count": int(last_row[3])
#             }
#             return jsonify(data)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/get_kpi_data')
def get_kpi_data():
    """Hardcoded KPI data for now"""
    return jsonify({
        "crowd_count": 35,   # Hardcoded crowd count
        "basket_count": 20,  # Hardcoded basket detection count
        "shoplifting_count": 0  # Hardcoded shoplifting incidents
    })


@app.route('/generate_ai_content', methods=['POST'])
def generate_content():
    """Call the GenAI function and return generated insights."""
    try:
        response_text = generate()  # Call the function directly
        return jsonify({"content": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,threaded=True)
