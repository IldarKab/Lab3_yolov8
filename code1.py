import random

import cv2
from ultralytics import YOLO

video_cap = cv2.VideoCapture('video.mp4')

out = cv2.VideoWriter('outpy.avi',
cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1920, 1080))

model = YOLO("yolov8n.pt")
colors = [[random.randint(0, 255) for i in range(3)] for name in model.names]
confidence_threshold = 0.7
while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    detections = model(frame)[0]
    for data in detections.boxes.data.tolist():  # Дата это прямоугольник и информации о том
        # что в нём хранится [xmin, ymin, xmas, ymax,confidence, class_id]
        xmin = int(data[0])
        ymin = int(data[1])
        xmax = int(data[2])
        ymax = int(data[3])
        confidence = data[4]
        class_name = detections.names[int(data[5])]
        class_id = int(data[5])
        if confidence < confidence_threshold:
            continue
        r, g, b = colors[class_id]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (r, g, b), 2)
        cv2.putText(frame, class_name, (xmin, ymin - 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color = (r, g, b), thickness=2)
    out.write(frame)
    cv2.imshow("video", frame)
    if cv2.waitKey(2) == ord('q') : break
video_cap.release()
out.release()
cv2.destroyAllWindows()
