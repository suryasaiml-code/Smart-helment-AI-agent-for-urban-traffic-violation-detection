import cv2
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os
import math

# ==============================
# LOAD SETTINGS
# ==============================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not found in .env")
    exit()

# General YOLO model
yolo = YOLO("yolo11n.pt")

# Roboflow helmet model
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

HELMET_MODEL = "motor-cycle-helmet/6"

# ==============================
# HELPER FUNCTIONS
# ==============================

def center(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )


def inside(point, box):
    x, y = point
    x1, y1, x2, y2 = box

    return x1 <= x <= x2 and y1 <= y <= y2


def distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("======================================")
print(" SMART HELMET VIOLATION TEST")
print("======================================")
print("Press Q to stop.")

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera frame could not be read.")
        break

    frame_number += 1

    # --------------------------------
    # YOLO DETECTION
    # --------------------------------

    results = yolo.track(
        frame,
        persist=True,
        verbose=False
    )

    motorcycles = []
    people = []

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < 0.40:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            # COCO class 0 = person
            if cls == 0:

                people.append({
                    "box": (x1, y1, x2, y2),
                    "id": track_id
                })

            # COCO class 3 = motorcycle
            elif cls == 3:

                motorcycles.append({
                    "box": (x1, y1, x2, y2),
                    "id": track_id
                })

    # --------------------------------
    # DRAW MOTORCYCLES
    # --------------------------------

    for motorcycle in motorcycles:

        x1, y1, x2, y2 = motorcycle["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "MOTORCYCLE",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # --------------------------------
    # DRAW PEOPLE
    # --------------------------------

    for person in people:

        x1, y1, x2, y2 = person["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2
        )

        label = "RIDER"

        if person["id"] is not None:
            label += f" #{person['id']}"

        cv2.putText(
            frame,
            label,
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

    # --------------------------------
    # HELMET MODEL
    #
    # Run every 5th frame to reduce
    # Roboflow requests.
    # --------------------------------

    helmet_predictions = []

    if frame_number % 5 == 0:

        try:

            result = CLIENT.infer(
                frame,
                model_id=HELMET_MODEL
            )

            helmet_predictions = result.get(
                "predictions",
                []
            )

        except Exception as e:

            print("Roboflow error:", e)

    # --------------------------------
    # VIOLATION ANALYSIS
    # --------------------------------

    violation_found = False
    triple_riding = False

    # For every motorcycle,
    # find nearby people.
    for motorcycle in motorcycles:

        motorcycle_box = motorcycle["box"]

        mx, my = center(motorcycle_box)

        associated_people = []

        for person in people:

            person_box = person["box"]

            px, py = center(person_box)

            # Distance-based association
            max_distance = max(
                motorcycle_box[2] - motorcycle_box[0],
                motorcycle_box[3] - motorcycle_box[1]
            ) * 1.5

            if distance(
                (mx, my),
                (px, py)
            ) < max_distance:

                associated_people.append(person)

        # --------------------------------
        # TRIPLE RIDING
        # --------------------------------

        if len(associated_people) >= 3:

            triple_riding = True
            violation_found = True

            cv2.putText(
                frame,
                "TRIPLE RIDING!",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        # --------------------------------
        # HELMET CHECK
        # --------------------------------

        for person in associated_people:

            person_box = person["box"]

            px1, py1, px2, py2 = person_box

            # Look only around the upper part
            # of the person for helmet detection.
            head_area = (
                px1,
                py1,
                px2,
                py1 + int((py2 - py1) * 0.40)
            )

            helmet_status = "UNKNOWN"

            for prediction in helmet_predictions:

                confidence = prediction["confidence"]

                if confidence < 0.50:
                    continue

                hx = prediction["x"]
                hy = prediction["y"]

                prediction_point = (
                    int(hx),
                    int(hy)
                )

                if inside(
                    prediction_point,
                    head_area
                ):

                    class_name = prediction["class"].lower()

                    if "without" in class_name:

                        helmet_status = "NO HELMET"

                    elif "helmet" in class_name:

                        helmet_status = "HELMET"

            # --------------------------------
            # NO HELMET VIOLATION
            # --------------------------------

            if helmet_status == "NO HELMET":

                violation_found = True

                cv2.putText(
                    frame,
                    "NO HELMET!",
                    (px1, max(py1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3
                )

            elif helmet_status == "HELMET":

                cv2.putText(
                    frame,
                    "HELMET",
                    (px1, max(py1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # --------------------------------
    # FINAL STATUS
    # --------------------------------

    if violation_found:

        status = "VIOLATION DETECTED"

        cv2.putText(
            frame,
            status,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    else:

        status = "NORMAL"

        cv2.putText(
            frame,
            status,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

    # --------------------------------
    # SHOW COUNTS
    # --------------------------------

    cv2.putText(
        frame,
        f"Motorcycles: {len(motorcycles)}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"People: {len(people)}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # DISPLAY
    # --------------------------------

    cv2.imshow(
        "Smart Helmet - Violation Detection",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("Violation test stopped.")