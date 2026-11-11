import cv2
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

# ==========================================
# SMART HELMET AI - REAL TIME
# ==========================================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not found in .env")
    exit()

# YOLO model
yolo = YOLO("yolo11n.pt")

# Roboflow helmet model
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

HELMET_MODEL = "motor-cycle-helmet/6"

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("===================================")
print(" SMART HELMET AI - LIVE")
print("===================================")
print("Camera started.")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # --------------------------------------
    # YOLO: PERSON + MOTORCYCLE
    # --------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated = results[0].plot()

    # --------------------------------------
    # SHOW DETECTION STATUS
    # --------------------------------------

    person_count = 0
    motorcycle_count = 0

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        if confidence < 0.40:
            continue

        if class_id == 0:
            person_count += 1

        elif class_id == 3:
            motorcycle_count += 1

    # --------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------

    cv2.putText(
        annotated,
        "SMART HELMET AI - LIVE",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Motorcycles: {motorcycle_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"People: {person_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        "LIVE",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # --------------------------------------
    # DISPLAY
    # --------------------------------------

    cv2.imshow(
        "Smart Helmet AI - Real Time",
        annotated
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Camera stopped.")