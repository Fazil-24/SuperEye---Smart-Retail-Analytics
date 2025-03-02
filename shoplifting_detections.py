import cv2
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers.legacy import Adam

# Load the Keras model
try:
    model = load_model('model.h5', compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define classes
classes = ['abnormal', 'normal']

# Load pre-trained MobileNet SSD for person detection
proto_path = "models/MobileNetSSD_deploy.prototxt"
model_path = "models/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def play_alert():
    """Play an audio alert when shoplifting is detected."""
    engine.say("Alert! Alert! Shoplifting detected")
    engine.runAndWait()

def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def classify_and_detect(frame):
    height, width = frame.shape[:2]
    
    processed_frame = preprocess_image(frame)
    predictions = model.predict(processed_frame, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence_person = detections[0, 0, i, 2]
        if confidence_person > 0.4 and int(detections[0, 0, i, 1]) == 15:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    label_text = f"Class: {predicted_class} (Confidence: {confidence:.2f})"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if predicted_class == "abnormal":
        cv2.putText(frame, "ALERT: Shoplifting", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # play_alert()

    return frame
