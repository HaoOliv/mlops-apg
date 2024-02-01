import sys
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import pandas as pd
import math

# Load the YOLO model
model = YOLO('./yolov8n.pt')

# Initialise a Flask object
app = Flask(__name__)
def health_check():
    # You can perform any health checks here
    # For example, checking the database connection, external service availability, etc.
    # For this example, let's assume everything is fine and return a simple JSON response
    return jsonify({'status': 'ok'})

@app.route('/predict')
def predict_image():
    # Get the image URL from the request
    image_url = request.args.get('image_url')

    # Download the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Make a prediction
    predictions = model(image, save_conf=True, conf = 0.2)
    results = []
    class_id = model.names
    for prediction in predictions:
        boxes = prediction.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls)
            results.append([x1,y1,x2,y2, confidence, class_id[int(cls)]])

    df = pd.DataFrame(results, columns=['x1', 'y1', 'x2', 'y2', 'Confidence', 'Class ID'])

    # return prediction result
    return df.to_json(orient='records')

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)
