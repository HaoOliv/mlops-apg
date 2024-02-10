from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO

def freeze_layer(trainer):
    model = YOLO('yolov8n_v1.pt')
    num_freeze = 5
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)] 
    for k, v in model.named_parameters():
        v.requires_grad = True 
    if any(x in k for x in freeze):
        print(f'freezing {k}')
        v.requires_grad = False
    print(f"{num_freeze} layers are freezed.")
model = YOLO('yolov8n_test.pt')
model.add_callback("on_train_start", freeze_layer)
dataset = './pothole_v8.yaml' 
folder_path = 'test_result' 
print('model loaded')
model.train(data = dataset, epochs = 10, batch = 8, pretrained = True, imgsz = 640)

model.val() 
print('val complete')

results = model.predict(source=folder_path, save=True, conf = 0.33, line_width = 1, imgsz = 640)
path = model.export()