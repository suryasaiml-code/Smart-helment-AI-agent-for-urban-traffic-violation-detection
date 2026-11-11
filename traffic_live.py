import cv2
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Get Roboflow API key
API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    print("ERROR: Roboflow API key not found.")
    exit()

# -----------------------------
# Load general YOLO model
# -----------------------------
yolo_model = YOLO("yolo11n.pt")

# -----------------------------
# Connect to Roboflow
# -----------------------------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

HELMET_MODEL = "motor-cycle-helmet/6"

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("===================================")
print(" SMART HELMET AI - LIVE DETECTION")
print("===================================")
print("Press Q to stop.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # =================================
    # 1. General YOLO detection
    # =================================
    yolo_results = yolo_model(frame, verbose=False)

    # Draw YOLO detections
    annotated_frame = yolo_results[0].plot()

    # =================================
    # 2. Helmet-specific Roboflow model
    # =================================
    try:

        result = CLIENT.infer(
            frame,
            model_id=HELMET_MODEL
        )

        predictions = result.get("predictions", [])

        for prediction in predictions:

            class_name = prediction["class"]
            confidence = prediction["confidence"]

            x = prediction["x"]
            y = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]

            # Convert center coordinates to corners
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            # Draw helmet detection
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{class_name} {confidence * 100:.1f}%"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    except Exception as e:

        cv2.putText(
            annotated_frame,
            "Roboflow detection error",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        print("Roboflow error:", e)

    # =================================
    # Show result
    # =================================
    cv2.imshow(
        "Smart Helmet AI - Live Traffic Detection",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Traffic detection stopped.")