from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model = YOLO("models/vin_detector.pt")

def detect_vin_crop(image_path: str):
    results = model.predicte(image_path, conf=0.25, imgsz=960)

    image = cv2.imread(image_path)
    best_box = None
    best_conf = 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

    if best_box is None:
        return None
    
    x1, y1, x2, y2 = map(int, best_box)

    pad = 20
    h, w = image.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = image[y1:y2, x1:x2]

    return crop

def preprocess_vin(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    return gray