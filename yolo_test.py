from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Test image
results = model("https://ultralytics.com/images/bus.jpg", show=True)

print("YOLO test completed!")