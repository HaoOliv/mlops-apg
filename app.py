import sys
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2
import requests
from io import BytesIO
from flask import Flask, request

# Load the YOLO model
model = YOLO('./yolov8n.pt')

# Initialise a Flask object
app = Flask(__name__)

@app.route('/predict')
def predict_image():
    # Get the image URL from the request
    image_url = request.args.get('image_url')

    # Download the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Make a prediction
    prediction = model(image)

    # Return the prediction
    return str(prediction[0])

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)