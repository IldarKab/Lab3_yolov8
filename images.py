
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

for i in range(1, 21):
    results = model(f'picture{i}.jpg', save=True)