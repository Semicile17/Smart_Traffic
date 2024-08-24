import cv2
from ultralytics import YOLO

# Load YOLOv8n pre-trained model
model = YOLO("yolov8n.pt")

# Load image
image_path = "E:/UPES/Traffic/test1.jpg"
frame = cv2.imread(image_path)

# Ensure the image is loaded
if frame is None:
    print(f"Error: Could not load image from path: {image_path}")
    exit()

# Resize the image to standard YOLO size (640x640)
frame_resized = cv2.resize(frame, (640, 640))

# Perform inference with the resized image
results = model(frame_resized)

# Check if any boxes were detected
if results[0].boxes.shape[0] == 0:
    print("No objects detected.")
else:
    # Print details about detected boxes
    print(results[0].boxes)
    
    # Visualize the detections
    results[0].plot()

    # Display the frame with detections
    cv2.imshow('YOLOv8 Detection', frame_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
