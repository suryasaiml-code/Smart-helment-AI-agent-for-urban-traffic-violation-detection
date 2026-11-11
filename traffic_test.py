import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Traffic detection started.")
print("Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # Detect objects
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow(
        "Smart Helmet - Traffic Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Traffic detection stopped.")