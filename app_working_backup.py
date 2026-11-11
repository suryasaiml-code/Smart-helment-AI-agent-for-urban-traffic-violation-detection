from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

import os
import cv2
import math
import tempfile


# ==========================================
# SETTINGS
# ==========================================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY not found in .env file"
    )


# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Smart Helmet AI",
    version="0.2.0"
)


# ==========================================
# LOAD MODELS
# ==========================================

# General object detection model
yolo = YOLO("yolo11n.pt")

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# Helmet detection model
HELMET_MODEL = "motor-cycle-helmet/6"


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
        (point1[0] - point2[0]) ** 2
        +
        (point1[1] - point2[1]) ** 2
    )


def point_inside(point, box):

    x, y = point

    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


# ==========================================
# HOME API
# ==========================================

@app.get("/")
def home():

    return {
        "status": "online",
        "system": "Smart Helmet AI",
        "message": "Backend is running"
    }


# ==========================================
# HEALTH API
# ==========================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "yolo_model": "loaded",
        "helmet_model": HELMET_MODEL
    }


# ==========================================
# DETECTION API
# ==========================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    # ======================================
    # SAVE UPLOADED IMAGE TEMPORARILY
    # ======================================

    contents = await file.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp:

        temp.write(contents)

        image_path = temp.name

    try:

        # ==================================
        # READ IMAGE
        # ==================================

        frame = cv2.imread(image_path)

        if frame is None:

            return {
                "success": False,
                "message": "Could not read image"
            }


        # ==================================
        # YOLO DETECTION
        # ==================================

        results = yolo.track(
            frame,
            persist=True,
            verbose=False
        )

        people = []

        motorcycles = []


        # ==================================
        # PROCESS YOLO DETECTIONS
        # ==================================

        for box in results[0].boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            # Ignore weak detections
            if confidence < 0.40:
                continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            track_id = None

            if box.id is not None:

                track_id = int(
                    box.id[0]
                )


            # ------------------------------
            # PERSON
            # ------------------------------

            if class_id == 0:

                people.append({

                    "id":
                        track_id,

                    "box":
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                    "confidence":
                        round(
                            confidence,
                            3
                        )
                })


            # ------------------------------
            # MOTORCYCLE
            # ------------------------------

            elif class_id == 3:

                motorcycles.append({

                    "id":
                        track_id,

                    "box":
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),

                    "confidence":
                        round(
                            confidence,
                            3
                        )
                })


        # ==================================
        # ROBOFLOW HELMET DETECTION
        # ==================================

        helmet_result = CLIENT.infer(
            image_path,
            model_id=HELMET_MODEL
        )


        all_predictions = (
            helmet_result.get(
                "predictions",
                []
            )
        )


        # Only keep helmet-related classes
        helmet_predictions = [

            prediction

            for prediction
            in all_predictions

            if (
                prediction["class"]
                .lower()
                in [
                    "helmet",
                    "withouthelmet"
                ]
            )
        ]


        # ==================================
        # RIDER ↔ MOTORCYCLE ASSOCIATION
        # ==================================
        #
        # IMPORTANT:
        # A person can belong to ONLY ONE
        # motorcycle.
        #
        # We assign each person to the
        # closest valid motorcycle.
        # ==================================

        motorcycle_riders = []


        # Create rider list for each bike
        for motorcycle in motorcycles:

            motorcycle_riders.append({

                "motorcycle":
                    motorcycle,

                "riders":
                    []
            })


        # Process each person once
        for person in people:

            person_center = center(
                person["box"]
            )


            best_motorcycle_index = None

            best_distance = float(
                "inf"
            )


            # ------------------------------
            # FIND CLOSEST MOTORCYCLE
            # ------------------------------

            for index, motorcycle in enumerate(
                motorcycles
            ):

                motorcycle_box = (
                    motorcycle["box"]
                )


                motorcycle_center = center(
                    motorcycle_box
                )


                motorcycle_width = (
                    motorcycle_box[2]
                    -
                    motorcycle_box[0]
                )


                motorcycle_height = (
                    motorcycle_box[3]
                    -
                    motorcycle_box[1]
                )


                # Search region around bike
                search_distance = max(
                    motorcycle_width,
                    motorcycle_height
                ) * 1.5


                d = distance(
                    person_center,
                    motorcycle_center
                )


                # Must be close enough
                # AND closer than previous bike
                if (
                    d <= search_distance
                    and
                    d < best_distance
                ):

                    best_distance = d

                    best_motorcycle_index = (
                        index
                    )


            # ------------------------------
            # ASSIGN PERSON ONLY ONCE
            # ------------------------------

            if (
                best_motorcycle_index
                is not None
            ):

                motorcycle_riders[
                    best_motorcycle_index
                ]["riders"].append({

                    "person":
                        person,

                    "distance":
                        round(
                            best_distance,
                            2
                        )
                })


        # ==================================
        # VIOLATION ANALYSIS
        # ==================================

        violations = []

        rider_results = []


        # ==================================
        # CHECK EACH MOTORCYCLE
        # ==================================

        for item in motorcycle_riders:

            motorcycle = (
                item["motorcycle"]
            )

            riders = (
                item["riders"]
            )

            rider_count = len(
                riders
            )


            # ==================================
            # TRIPLE RIDING CHECK
            # ==================================

            if rider_count >= 3:

                violations.append({

                    "motorcycle_id":
                        motorcycle["id"],

                    "type":
                        "TRIPLE RIDING",

                    "rider_count":
                        rider_count
                })


            # ==================================
            # CHECK EACH RIDER
            # ==================================

            for rider_data in riders:

                person = (
                    rider_data["person"]
                )


                person_box = (
                    person["box"]
                )


                px1, py1, px2, py2 = (
                    person_box
                )


                person_height = (
                    py2 - py1
                )


                # ==================================
                # APPROXIMATE HEAD REGION
                # ==================================

                head_box = (

                    px1,

                    py1,

                    px2,

                    py1
                    +
                    int(
                        person_height
                        * 0.45
                    )
                )


                rider_status = (
                    "UNKNOWN"
                )


                best_confidence = 0


                # ==================================
                # MATCH HELMET TO RIDER HEAD
                # ==================================

                for prediction in (
                    helmet_predictions
                ):

                    class_name = (
                        prediction["class"]
                        .lower()
                    )


                    confidence = float(
                        prediction[
                            "confidence"
                        ]
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


                    # Helmet detection must
                    # fall inside rider head
                    if point_inside(
                        helmet_point,
                        head_box
                    ):

                        if (
                            confidence
                            >
                            best_confidence
                        ):

                            best_confidence = (
                                confidence
                            )


                            if (
                                class_name
                                ==
                                "helmet"
                            ):

                                rider_status = (
                                    "HELMET"
                                )


                            elif (
                                class_name
                                ==
                                "withouthelmet"
                            ):

                                rider_status = (
                                    "NO HELMET"
                                )


                # ==================================
                # SAVE RIDER RESULT
                # ==================================

                rider_results.append({

                    "motorcycle_id":
                        motorcycle["id"],

                    "rider_id":
                        person["id"],

                    "helmet_status":
                        rider_status,

                    "confidence":
                        round(
                            best_confidence,
                            3
                        )
                })


                # ==================================
                # NO HELMET VIOLATION
                # ==================================

                if (
                    rider_status
                    ==
                    "NO HELMET"
                ):

                    violations.append({

                        "motorcycle_id":
                            motorcycle["id"],

                        "rider_id":
                            person["id"],

                        "type":
                            "NO HELMET",

                        "confidence":
                            round(
                                best_confidence,
                                3
                            )
                    })


        # ==================================
        # CLEAN HELMET RESULTS
        # ==================================

        clean_helmet_results = []


        for prediction in helmet_predictions:

            clean_helmet_results.append({

                "class":
                    prediction["class"],

                "confidence":
                    round(
                        float(
                            prediction[
                                "confidence"
                            ]
                        ),
                        3
                    )
            })


        # ==================================
        # RETURN WEBSITE-READY JSON
        # ==================================

        return {

            "success":
                True,

            "motorcycles_detected":
                len(
                    motorcycles
                ),

            "people_detected":
                len(
                    people
                ),

            "associated_riders":
                len(
                    rider_results
                ),

            "riders":
                rider_results,

            "helmet_detections":
                clean_helmet_results,

            "violation_count":
                len(
                    violations
                ),

            "violations":
                violations
        }


    # ======================================
    # DELETE TEMPORARY IMAGE
    # ======================================

    finally:

        if os.path.exists(
            image_path
        ):

            os.remove(
                image_path
            )