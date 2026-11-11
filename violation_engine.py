import cv2
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
import os
import math

# ==========================================
# SETTINGS
# ==========================================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not found in .env")
    exit()

IMAGE_PATH = "bike.jpg"

# ==========================================
# LOAD MODELS
# ==========================================

yolo = YOLO("yolo11n.pt")

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

HELMET_MODEL = "motor-cycle-helmet/6"

# ==========================================
# READ IMAGE
# ==========================================

frame = cv2.imread(IMAGE_PATH)

if frame is None:
    print("ERROR: bike.jpg not found.")
    exit()

# ==========================================
# YOLO DETECTION
# ==========================================

results = yolo.track(
    frame,
    persist=True,
    verbose=False
)

people = []
motorcycles = []

for box in results[0].boxes:

    class_id = int(box.cls[0])
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

    # Person
    if class_id == 0:

        people.append({
            "id": track_id,
            "box": (x1, y1, x2, y2)
        })

    # Motorcycle
    elif class_id == 3:

        motorcycles.append({
            "id": track_id,
            "box": (x1, y1, x2, y2)
        })

# ==========================================
# ROBOFLOW HELMET DETECTION
# ==========================================

print("\nRunning helmet detection...")

helmet_result = CLIENT.infer(
    IMAGE_PATH,
    model_id=HELMET_MODEL
)

all_predictions = helmet_result.get(
    "predictions",
    []
)

helmet_predictions = [
    p for p in all_predictions
    if p["class"].lower() in [
        "helmet",
        "withouthelmet"
    ]
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )


def distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def point_inside(point, box):

    x, y = point

    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2 and
        y1 <= y <= y2
    )

# ==========================================
# ONE-TO-ONE RIDER ASSOCIATION
# ==========================================

print("\n===================================")
print(" RIDER ASSOCIATION")
print("===================================")

motorcycle_riders = []

# Store every possible motorcycle-person pair
possible_pairs = []

for motorcycle in motorcycles:

    motorcycle_center = center(
        motorcycle["box"]
    )

    motorcycle_width = (
        motorcycle["box"][2] -
        motorcycle["box"][0]
    )

    motorcycle_height = (
        motorcycle["box"][3] -
        motorcycle["box"][1]
    )

    max_distance = max(
        motorcycle_width,
        motorcycle_height
    ) * 1.5

    for person in people:

        person_center = center(
            person["box"]
        )

        d = distance(
            motorcycle_center,
            person_center
        )

        if d <= max_distance:

            possible_pairs.append({
                "motorcycle": motorcycle,
                "person": person,
                "distance": d
            })


# ==========================================
# ASSIGN EACH RIDER TO ONLY ONE MOTORCYCLE
# ==========================================

used_motorcycles = set()
used_people = set()

# Closest motorcycle-person pairs first
possible_pairs.sort(
    key=lambda x: x["distance"]
)

for pair in possible_pairs:

    motorcycle = pair["motorcycle"]
    person = pair["person"]

    motorcycle_id = motorcycle["id"]
    person_id = person["id"]

    if motorcycle_id in used_motorcycles:
        continue

    if person_id in used_people:
        continue

    used_motorcycles.add(motorcycle_id)
    used_people.add(person_id)

    motorcycle_riders.append({
        "motorcycle": motorcycle,
        "riders": [
            {
                "person": person,
                "distance": pair["distance"]
            }
        ]
    })

    print(
        f"Motorcycle ID {motorcycle_id} "
        f"→ Rider ID {person_id}"
    )


# ==========================================
# MOTORCYCLES WITHOUT ASSIGNED RIDER
# ==========================================

for motorcycle in motorcycles:

    motorcycle_id = motorcycle["id"]

    already_assigned = any(
        item["motorcycle"]["id"] == motorcycle_id
        for item in motorcycle_riders
    )

    if not already_assigned:

        motorcycle_riders.append({
            "motorcycle": motorcycle,
            "riders": []
        })

        print(
            f"Motorcycle ID {motorcycle_id} "
            f"→ No rider assigned"
        )

# ==========================================
# TRIPLE RIDING CHECK
# ==========================================

print("\n===================================")
print(" TRIPLE RIDING ANALYSIS")
print("===================================")

violations = []

for item in motorcycle_riders:

    motorcycle = item["motorcycle"]
    riders = item["riders"]

    rider_count = len(riders)

    if rider_count >= 3:

        print(
            f"🚨 TRIPLE RIDING DETECTED"
            f" - Motorcycle ID "
            f"{motorcycle['id']}"
        )

        print(
            f"Rider count: {rider_count}"
        )

        violations.append({
            "motorcycle_id":
                motorcycle["id"],

            "type":
                "TRIPLE RIDING",

            "rider_count":
                rider_count
        })

    else:

        print(
            f"Motorcycle ID "
            f"{motorcycle['id']} "
            f"→ Rider count: "
            f"{rider_count}"
        )

# ==========================================
# HELMET ANALYSIS
# ==========================================

print("\n===================================")
print(" HELMET ANALYSIS")
print("===================================")

helmet_violations = []

for item in motorcycle_riders:

    motorcycle = item["motorcycle"]

    riders = item["riders"]

    for rider_data in riders:

        person = rider_data["person"]

        person_box = person["box"]

        px1, py1, px2, py2 = person_box

        person_height = py2 - py1

        # Upper 45% = approximate head region
        head_box = (
            px1,
            py1,
            px2,
            py1 + int(
                person_height * 0.45
            )
        )

        rider_status = "UNKNOWN"

        best_confidence = 0

        for prediction in helmet_predictions:

            class_name = (
                prediction["class"]
                .lower()
            )

            confidence = float(
                prediction["confidence"]
            )

            hx = float(
                prediction["x"]
            )

            hy = float(
                prediction["y"]
            )

            helmet_point = (
                int(hx),
                int(hy)
            )

            if point_inside(
                helmet_point,
                head_box
            ):

                if confidence > best_confidence:

                    best_confidence = confidence

                    if class_name == "helmet":

                        rider_status = "HELMET"

                    elif class_name == "withouthelmet":

                        rider_status = "NO HELMET"

        print(
            f"Motorcycle "
            f"{motorcycle['id']} "
            f"→ Rider "
            f"{person['id']} "
            f"→ {rider_status}"
        )

        if rider_status == "NO HELMET":

            helmet_violations.append({
                "motorcycle_id":
                    motorcycle["id"],

                "person_id":
                    person["id"],

                "type":
                    "NO HELMET",

                "confidence":
                    best_confidence
            })

# ==========================================
# FINAL VIOLATION SUMMARY
# ==========================================

print("\n===================================")
print(" FINAL VIOLATION SUMMARY")
print("===================================")

all_violations = (
    violations +
    helmet_violations
)

if len(all_violations) == 0:

    print("No confirmed violations.")

else:

    for violation in all_violations:

        if violation["type"] == "NO HELMET":

            print(
                f"🚨 NO HELMET | "
                f"Motorcycle "
                f"{violation['motorcycle_id']} | "
                f"Rider "
                f"{violation['person_id']} | "
                f"Confidence "
                f"{violation['confidence'] * 100:.2f}%"
            )

        elif violation["type"] == "TRIPLE RIDING":

            print(
                f"🚨 TRIPLE RIDING | "
                f"Motorcycle "
                f"{violation['motorcycle_id']} | "
                f"Riders "
                f"{violation['rider_count']}"
            )

# ==========================================
# DRAW RESULTS
# ==========================================

annotated = results[0].plot()

# ==========================================
# DRAW HELMET BOXES
# ==========================================

for prediction in helmet_predictions:

    class_name = prediction["class"]

    confidence = float(
        prediction["confidence"]
    )

    x = int(prediction["x"])
    y = int(prediction["y"])

    width = int(prediction["width"])
    height = int(prediction["height"])

    x1 = int(x - width / 2)
    y1 = int(y - height / 2)

    x2 = int(x + width / 2)
    y2 = int(y + height / 2)

    if "without" in class_name.lower():

        box_color = (0, 0, 255)

    else:

        box_color = (0, 255, 0)

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    label = (
        f"{class_name} "
        f"{confidence * 100:.1f}%"
    )

    cv2.putText(
        annotated,
        label,
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2
    )

# ==========================================
# SAVE RESULT
# ==========================================

output = "violation_engine_result.jpg"

cv2.imwrite(
    output,
    annotated
)

print("\n===================================")
print(" RESULT")
print("===================================")

print(
    "Result saved as:",
    output
)