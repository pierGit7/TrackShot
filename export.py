from ultralytics import YOLO

model = YOLO("checkpoints/best.pt")
model.export(format="tflite", imgsz=96, int8=True, data="configs/data/data.yaml")
