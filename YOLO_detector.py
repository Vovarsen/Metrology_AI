from ultralytics import YOLO
import numpy as np


def f(model_path, img_path):
    model = YOLO(model_path)
    result = model(img_path)[0]
    box = result.boxes.xywh.cpu().numpy().astype(np.int32)
    x1, y1, w, h = box[0]
    x1 = int(x1-w/2)
    y1 = int(y1-h/2)
    return (x1, y1, w, h)


#MODEL_NAME = "best.pt"
#img = "C:/Users/arsen/OneDrive/Pictures/car_dataset/frame_000000.PNG"
#print(f(MODEL_NAME, img))
