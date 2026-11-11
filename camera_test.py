import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show camera
    cv2.imshow("Smart Helmet AI - YOLO Test", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()