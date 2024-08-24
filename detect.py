import cv2
from ultralytics import YOLO

# Load the YOLO models
vehicle_model = YOLO("E:/UPES/Traffic/vehicle.pt", task="detect")  # Vehicle detection model
ambulance_model = YOLO("E:/UPES/Traffic/ambulance.pt", task="detect")  # Ambulance detection model

# Function to draw bounding boxes and labels
def draw_boxes(detections, frame, label, color):
    for result in detections:
        detection_count = result.boxes.shape[0]
        for i in range(detection_count):
            cls = int(result.boxes.cls[i].item())
            confidence = float(result.boxes.conf[i].item())
            bounding_box = result.boxes.xyxy[i].cpu().numpy()

            x1, y1, x2, y2 = map(int, bounding_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Open video capture (0 is for webcam or replace with video path for file input)
video_path = "E:/UPES/Traffic/video1.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video is opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video from path: {video_path}")
    exit()

# Loop through video frames
while True:
    ret, frame = cap.read()

    # Check if the frame is successfully captured
    if not ret:
        print("End of video or error capturing frame.")
        break

    # Use the vehicle model to detect vehicles
    vehicle_results = vehicle_model(frame)

    # Use the ambulance model to detect ambulances
    ambulance_results = ambulance_model(frame)

    # Draw bounding boxes for ambulances (yellow color)
    if ambulance_results and ambulance_results[0].boxes:
        draw_boxes(ambulance_results, frame, 'ambulance', (0, 255, 255))  # Yellow box for ambulance

    # Draw bounding boxes for vehicles (green color)
    if vehicle_results and vehicle_results[0].boxes:
        draw_boxes(vehicle_results, frame, 'vehicle', (0, 255, 0))  # Green box for vehicles

    # Display the frame with detections
    cv2.imshow('Ambulance and Vehicle Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
