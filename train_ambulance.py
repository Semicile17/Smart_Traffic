from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

model.train(data="data_ambulance.yaml", epochs=6) 

model.conf = 0.5  # Example confidence threshold
model.iou = 0.4   # Example IoU threshold for NMS
model.max_det = 100

path = model.export(format="onnx") 

print(path)


