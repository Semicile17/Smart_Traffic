from ultralytics import YOLO

# Load and verify the vehicle detection model
vehicle_model = YOLO("E:/UPES/Traffic/vehicle.pt")

# Print the model summary to check if it’s loaded properly
print(vehicle_model)
