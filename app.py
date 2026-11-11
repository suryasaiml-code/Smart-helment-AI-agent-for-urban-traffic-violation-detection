from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
import os
import cv2
import math
import tempfile
import requests
import shutil
import logging
import re
from threading import Lock
from starlette.concurrency import run_in_threadpool
from datetime import datetime
from pathlib import Path
from database import (
    save_violation,
    get_all_violations,
    get_violation,
    get_statistics,
    delete_violation
)
# ============================================================
# SETTINGS
# ============================================================
load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY not found in .env file")
# ============================================================
# PATH SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)
# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Smart Helmet AI",
    version="0.6.0"
)
# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://smart-helmet-ai-traf-q6b5.bolt.host",

    "http://localhost:5173",
    "http://127.0.0.1:5173",

    "http://localhost:5174",
    "http://127.0.0.1:5174",

    "http://localhost:5175",
    "http://127.0.0.1:5175",

    "http://localhost:5176",
    "http://127.0.0.1:5176",

    "http://localhost:3000",
    "http://127.0.0.1:3000",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# EVIDENCE STATIC FILES
# ============================================================
app.mount(
    "/evidence",
    StaticFiles(directory=str(EVIDENCE_DIR)),
    name="evidence"
)
# ============================================================
# LOAD AI MODELS
# ============================================================
logger = logging.getLogger("smart_helmet")
MODEL_LOCK = Lock()
yolo = YOLO(os.getenv("YOLO_MODEL_PATH", "yolo11n.pt"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))
HELMET_CONF = float(os.getenv("HELMET_CONF", "0.40"))
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)
HELMET_MODEL = "motor-cycle-helmet/6"
# ============================================================
# OCR SERVICE
# ============================================================
OCR_SERVICE_URL = "http://127.0.0.1:8001/ocr"
# ============================================================
# HELPER FUNCTIONS
# ============================================================
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

def assign_frame_ids(objects):
    """Keep valid tracker IDs; allocate collision-free IDs in spatial order.

    Fallback IDs identify this frame only, not the same vehicle across uploads.
    """
    objects.sort(key=lambda item: (item["box"][0], item["box"][1], item["box"][2], item["box"][3]))
    used = {item["id"] for item in objects if item["id"] is not None}
    assigned = set()
    next_id = 1
    for item in objects:
        if item["id"] is None or item["id"] in assigned:
            while next_id in used or next_id in assigned:
                next_id += 1
            item["id"] = next_id
        assigned.add(item["id"])
    return objects


def detect_objects(frame):
    # Independent image requests must not inherit another camera's tracker.
    with MODEL_LOCK:
        results = yolo.predict(frame, classes=[0, 3], conf=YOLO_CONF,
                               imgsz=YOLO_IMGSZ, iou=0.5, verbose=False)
    people, motorcycles = [], []
    height, width = frame.shape[:2]
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coords = [float(v) for v in box.xyxy[0]]
            if not all(math.isfinite(v) for v in coords):
                continue
            x1, y1, x2, y2 = coords
            bounds = (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            track_id = int(box.id[0]) if box.id is not None else None
            item = {"id": track_id, "box": bounds, "confidence": round(confidence, 3)}
            if class_id == 0:
                people.append(item)
            elif class_id == 3:
                motorcycles.append(item)
    return assign_frame_ids(people), assign_frame_ids(motorcycles)


def rider_score(person_box, motorcycle_box):
    """
    Calculate how likely a detected person is riding a motorcycle.

    Higher score = stronger rider/motorcycle association.
    Returns None when the person is clearly unrelated.
    """

    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = motorcycle_box

    person_width = max(px2 - px1, 1)
    person_height = max(py2 - py1, 1)

    motorcycle_width = max(mx2 - mx1, 1)
    motorcycle_height = max(my2 - my1, 1)

    person_center_x = (px1 + px2) / 2
    person_center_y = (py1 + py2) / 2

    motorcycle_center_x = (mx1 + mx2) / 2
    motorcycle_center_y = (my1 + my2) / 2

    # -------------------------------------------------
    # EXPANDED MOTORCYCLE RIDER ZONE
    # -------------------------------------------------

    # Riders can extend considerably above the actual
    # motorcycle bounding box, especially with 2/3 riders.
    expanded_left = mx1 - motorcycle_width * 0.70
    expanded_right = mx2 + motorcycle_width * 0.70

    expanded_top = my1 - motorcycle_height * 2.20
    expanded_bottom = my2 + motorcycle_height * 0.35

    # Person must overlap the horizontal rider zone.
    horizontal_overlap = max(
        0,
        min(px2, expanded_right)
        - max(px1, expanded_left)
    )

    horizontal_overlap_ratio = (
        horizontal_overlap / person_width
    )

    if horizontal_overlap_ratio < 0.20:
        return None

    # Person centre should be inside/close to rider zone.
    if not (
        expanded_left <= person_center_x <= expanded_right
        and expanded_top <= person_center_y <= expanded_bottom
    ):
        return None

    # -------------------------------------------------
    # DISTANCE
    # -------------------------------------------------

    dx = (
        person_center_x
        - motorcycle_center_x
    )

    dy = (
        person_center_y
        - motorcycle_center_y
    )

    distance_value = (
        (dx ** 2 + dy ** 2) ** 0.5
    )

    max_distance = max(
        motorcycle_width * 2.0,
        motorcycle_height * 2.8
    )

    if distance_value > max_distance:
        return None

    normalized_distance = (
        distance_value / max_distance
    )

    # -------------------------------------------------
    # PERSON BOTTOM RELATION
    # -------------------------------------------------

    # A rider normally extends toward/over the motorcycle.
    vertical_gap = abs(
        py2 - motorcycle_center_y
    )

    vertical_score = max(
        0.0,
        1.0
        - (
            vertical_gap
            / max(
                motorcycle_height * 2.5,
                1
            )
        )
    )

    # -------------------------------------------------
    # FINAL SCORE
    # -------------------------------------------------

    score = (
        horizontal_overlap_ratio * 3.0
        +
        (1.0 - normalized_distance) * 2.0
        +
        vertical_score
    )

    return score


def associate_riders(people, motorcycles):
    """
    Associate every detected person with the most likely
    motorcycle.

    Multiple people are intentionally allowed to belong
    to the same motorcycle so triple riding can be detected.
    """

    associations = []

    for motorcycle in motorcycles:
        associations.append({
            "motorcycle": motorcycle,
            "riders": []
        })

    if not people or not motorcycles:
        return associations

    for person in people:

        best_motorcycle_index = None
        best_score = None

        for index, motorcycle in enumerate(motorcycles):

            score = rider_score(
                person["box"],
                motorcycle["box"]
            )

            if score is None:
                continue

            if (
                best_score is None
                or score > best_score
            ):
                best_score = score
                best_motorcycle_index = index

        if best_motorcycle_index is not None:

            associations[
                best_motorcycle_index
            ]["riders"].append({
                "person": person,
                "association_score": round(
                    best_score,
                    3
                )
            })

    return associations


def match_heads(groups, predictions):
    edges = []
    for group in groups:
        for rider in group["riders"]:
            person = rider["person"]
            x1, y1, x2, y2 = person["box"]
            pw, ph = x2 - x1, y2 - y1
            head = (x1 - 0.08 * pw, y1 - 0.05 * ph, x2 + 0.08 * pw, y1 + 0.35 * ph)
            for index, prediction in enumerate(predictions):
                try:
                    confidence = float(prediction["confidence"])
                    hx, hy = float(prediction["x"]), float(prediction["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) for v in (confidence, hx, hy)):
                    continue
                if confidence < HELMET_CONF or not point_inside((hx, hy), head):
                    continue
                offset = abs(hx - (x1 + x2) / 2) / pw + abs(hy - (y1 + 0.13 * ph)) / ph
                edges.append((confidence - 0.2 * offset, confidence, person["id"], index))
    matches, used_predictions = {}, set()
    for _, confidence, person_id, index in sorted(edges, reverse=True):
        if person_id in matches or index in used_predictions:
            continue
        status = "HELMET" if predictions[index]["class"] == "helmet" else "NO HELMET"
        matches[person_id] = (status, confidence)
        used_predictions.add(index)
    return matches


# ============================================================
# OCR FUNCTION
# ============================================================
def run_plate_ocr(image_path):
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                OCR_SERVICE_URL,
                files={
                    "file": (
                        "violation.jpg",
                        image_file,
                        "image/jpeg"
                    )
                },
                timeout=30
            )
        if response.status_code != 200:
            return {
                "success": False,
                "status": "OCR_SERVICE_ERROR",
                "confirmed_vehicle_number": None,
                "plate_candidates": [],
                "message": (
                    f"OCR service returned "
                    f"HTTP {response.status_code}"
                )
            }
        data = response.json()
        candidates = data.get(
            "plate_candidates",
            []
        )
        if len(candidates) == 0:
            status = "NO_PLATE_DETECTED"
        elif len(candidates) == 1:
            status = "SINGLE_CANDIDATE"
        else:
            status = "MULTIPLE_CANDIDATES"
        return {
            "success": True,
            "status": status,
            "confirmed_vehicle_number": None,
            "ocr_best_candidate": data.get(
                "vehicle_number"
            ),
            "ocr_best_confidence": data.get(
                "plate_confidence"
            ),
            "plate_candidates": candidates,
            "all_detected_text": data.get(
                "all_detected_text",
                []
            ),
            "message": (
                "OCR completed. Vehicle number "
                "remains unconfirmed until "
                "plate-to-motorcycle association "
                "is implemented."
            )
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status": "OCR_TIMEOUT",
            "confirmed_vehicle_number": None,
            "plate_candidates": [],
            "message": "OCR service timed out"
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status": "OCR_OFFLINE",
            "confirmed_vehicle_number": None,
            "plate_candidates": [],
            "message": (
                "Could not connect to "
                "OCR service on port 8001"
            )
        }
    except Exception as error:
        return {
            "success": False,
            "status": "OCR_ERROR",
            "confirmed_vehicle_number": None,
            "plate_candidates": [],
            "message": str(error)
        }
# ============================================================
# HOME API
# ============================================================
@app.get("/")
def home():
    return {
        "status": "online",
        "system": "Smart Helmet AI",
        "version": "0.6.0",
        "message": "Backend is running"
    }
# ============================================================
# HEALTH API
# ============================================================
@app.get("/health")
def health():
    ocr_online = False
    try:
        response = requests.get(
            "http://127.0.0.1:8001/",
            timeout=3
        )
        ocr_online = (
            response.status_code == 200
        )
    except Exception:
        ocr_online = False
    return {
        "success": True,
        "main_backend": "online",
        "database": "online",
        "yolo_model": "loaded",
        "helmet_model": HELMET_MODEL,
        "ocr_service": (
            "online"
            if ocr_online
            else "offline"
        )
    }
# ============================================================
# GET ALL VIOLATIONS
# ============================================================
@app.get("/violations")
def violations_history():
    data = get_all_violations()
    return {
        "success": True,
        "count": len(data),
        "violations": data
    }
# ============================================================
# GET ONE VIOLATION
# ============================================================
@app.get("/violations/{violation_id}")
def violation_details(
    violation_id: int
):
    data = get_violation(
        violation_id
    )
    if data is None:
        raise HTTPException(
            status_code=404,
            detail="Violation not found"
        )
    return {
        "success": True,
        "violation": data
    }
# ============================================================
# DELETE VIOLATION
# ============================================================
@app.delete("/violations/{violation_id}")
def remove_violation(violation_id: int):
    violation = get_violation(violation_id)
    if violation is None:
        raise HTTPException(
            status_code=404,
            detail="Violation not found"
        )
    evidence_image = violation.get("evidence_image")
    deleted_violation = delete_violation(violation_id)
    evidence_deleted = False
    if evidence_image:
        evidence_path = (EVIDENCE_DIR / evidence_image).resolve()
        shared = any(row.get("evidence_image") == evidence_image for row in get_all_violations())
        if evidence_path.parent == EVIDENCE_DIR.resolve() and not shared:
            try:
                if evidence_path.is_file():
                    evidence_path.unlink()
                    evidence_deleted = True
            except OSError:
                logger.warning("Could not delete evidence for violation %s", violation_id)
    return {
        "success": True,
        "message": "Violation deleted successfully",
        "violation_id": violation_id,
        "evidence_deleted": evidence_deleted,
        "deleted_violation": deleted_violation
    }
# ============================================================
# STATISTICS
# ============================================================
@app.get("/statistics")
def statistics():
    stats = get_statistics()
    return {
        "success": True,
        **stats
    }
# ============================================================
# HOTSPOT MAP API
# ============================================================
@app.get("/hotspots")
def hotspots():
    violations = get_all_violations()
    hotspot_records = []
    for violation in violations:
        latitude = violation.get(
            "latitude"
        )
        longitude = violation.get(
            "longitude"
        )
        # Ignore old violations without GPS data
        if (
            latitude is None
            or longitude is None
        ):
            continue
        hotspot_records.append({
            "id": violation.get("id"),
            "detected_at": violation.get(
                "detected_at"
            ),
            "date": violation.get(
                "date"
            ),
            "time": violation.get(
                "time"
            ),
            "violation_type": violation.get(
                "violation_type"
            ),
            "motorcycle_id": violation.get(
                "motorcycle_id"
            ),
            "rider_id": violation.get(
                "rider_id"
            ),
            "rider_count": violation.get(
                "rider_count"
            ),
            "confidence": violation.get(
                "confidence"
            ),
            "vehicle_number": violation.get(
                "vehicle_number"
            ),
            "vehicle_number_status": violation.get(
                "vehicle_number_status"
            ),
            "plate_candidates": violation.get(
                "plate_candidates"
            ) or [],
            "evidence_image": violation.get(
                "evidence_image"
            ),
            "location": violation.get(
                "location"
            ),
            "latitude": latitude,
            "longitude": longitude,
            "camera_id": violation.get(
                "camera_id"
            ),
            "status": violation.get(
                "status"
            )
        })
    return {
        "success": True,
        "count": len(hotspot_records),
        "hotspots": hotspot_records
    }
# ============================================================
# DETECTION API
# ============================================================
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    latitude: float | None = Form(
        default=None
    ),
    longitude: float | None = Form(
        default=None
    ),
    location: str | None = Form(
        default=None
    ),
    camera_id: str | None = Form(
        default=None
    )
):
    # --------------------------------------------------------
    # READ UPLOADED FILE
    # --------------------------------------------------------
    contents = await file.read()
    suffix = os.path.splitext(
        file.filename or ""
    )[1]
    if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    if not contents:
        raise HTTPException(400, "Uploaded image is empty")
    if latitude is not None and (not math.isfinite(latitude) or not -90 <= latitude <= 90):
        raise HTTPException(422, "Invalid latitude")
    if longitude is not None and (not math.isfinite(longitude) or not -180 <= longitude <= 180):
        raise HTTPException(422, "Invalid longitude")
    # --------------------------------------------------------
    # TEMPORARY IMAGE
    # --------------------------------------------------------
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:
        temp.write(contents)
        image_path = temp.name
    try:
        # ====================================================
        # DATE AND TIME
        # ====================================================
        detection_time = datetime.now()
        detection_timestamp = (
            detection_time.isoformat(
                timespec="seconds"
            )
        )
        detection_date = (
            detection_time.strftime(
                "%Y-%m-%d"
            )
        )
        detection_clock_time = (
            detection_time.strftime(
                "%H:%M:%S"
            )
        )
        # ====================================================
        # READ IMAGE
        # ====================================================
        frame = cv2.imread(
            image_path
        )
        if frame is None:
            return {
                "success": False,
                "message": "Could not read image"
            }
        people, motorcycles = await run_in_threadpool(detect_objects, frame)
        helmet_result = await run_in_threadpool(
            CLIENT.infer, image_path, model_id=HELMET_MODEL
        )
        helmet_predictions = []
        for prediction in helmet_result.get("predictions", []):
            class_name = re.sub(r"[^a-z]", "", str(prediction.get("class", "")).lower())
            if class_name in {"helmet", "withhelmet", "withouthelmet", "nohelmet"}:
                prediction = dict(prediction)
                prediction["class"] = "helmet" if class_name in {"helmet", "withhelmet"} else "withouthelmet"
                helmet_predictions.append(prediction)
        motorcycle_riders = associate_riders(people, motorcycles)
        head_matches = match_heads(motorcycle_riders, helmet_predictions)
        # ====================================================
        # VIOLATION ANALYSIS
        # ====================================================
        violations = []
        rider_results = []
        # ====================================================
        # CHECK MOTORCYCLES
        # ====================================================
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
            # ------------------------------------------------
            # TRIPLE RIDING
            # ------------------------------------------------
            if rider_count >= 3:
                violations.append({
                    "motorcycle_id":
                        motorcycle["id"],
                    "type":
                        "TRIPLE RIDING",
                    "rider_count":
                        rider_count,
                    "confidence":
                        None
                })
            # =================================================
            # CHECK EACH RIDER
            # =================================================
            for rider_data in riders:
                person = (
                    rider_data["person"]
                )
                rider_status, best_confidence = head_matches.get(person["id"], ("UNKNOWN", 0.0))
                # ------------------------------------------------
                # SAVE RIDER RESULT
                # ------------------------------------------------
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
                # ------------------------------------------------
                # NO HELMET VIOLATION
                # ------------------------------------------------
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
                        "rider_count":
                            rider_count,
                        "confidence":
                            round(
                                best_confidence,
                                3
                            )
                    })
        # ====================================================
        # CLEAN HELMET RESULTS
        # ====================================================
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
        # ====================================================
        # NUMBER PLATE OCR
        # ====================================================
        if len(violations) > 0:
            plate_ocr = await run_in_threadpool(run_plate_ocr, image_path)
        else:
            plate_ocr = {
                "success": True,
                "status":
                    "NOT_REQUIRED",
                "confirmed_vehicle_number":
                    None,
                "plate_candidates":
                    [],
                "message": (
                    "OCR was not run because "
                    "no violation was detected."
                )
            }
        # ====================================================
        # SAVE EVIDENCE IMAGE
        # ====================================================
        evidence_filename = None
        evidence_path = None
        evidence_url = None
        if len(violations) > 0:
            evidence_filename = (
                "violation_"
                +
                detection_time.strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                +
                suffix
            )
            evidence_path = (
                EVIDENCE_DIR
                /
                evidence_filename
            )
            shutil.copy2(
                image_path,
                evidence_path
            )
            evidence_url = (
                f"/evidence/"
                f"{evidence_filename}"
            )
        # ====================================================
        # OCR PLATE CANDIDATES
        # ====================================================
        plate_candidates = (
            plate_ocr.get(
                "plate_candidates",
                []
            )
        )
        # ====================================================
        # SAVE VIOLATIONS TO SQLITE
        # ====================================================
        saved_violation_ids = []
        for violation in violations:
            violation[
                "detected_at"
            ] = detection_timestamp
            violation[
                "date"
            ] = detection_date
            violation[
                "time"
            ] = detection_clock_time
            violation[
                "vehicle_number"
            ] = None
            violation[
                "vehicle_number_status"
            ] = "UNCONFIRMED"
            # GPS/location information is also returned
            # with each violation.
            violation["location"] = location
            violation["latitude"] = latitude
            violation["longitude"] = longitude
            violation["camera_id"] = camera_id
            violation_id = save_violation(
                detected_at=
                    detection_timestamp,
                date=
                    detection_date,
                time=
                    detection_clock_time,
                violation_type=
                    violation["type"],
                motorcycle_id=
                    violation.get(
                        "motorcycle_id"
                    ),
                rider_id=
                    violation.get(
                        "rider_id"
                    ),
                rider_count=
                    violation.get(
                        "rider_count"
                    ),
                confidence=
                    violation.get(
                        "confidence"
                    ),
                vehicle_number=
                    None,
                vehicle_number_status=
                    "UNCONFIRMED",
                plate_candidates=
                    plate_candidates,
                evidence_image=
                    evidence_filename,
                location=
                    location,
                latitude=
                    latitude,
                longitude=
                    longitude,
                camera_id=
                    camera_id
            )
            violation[
                "violation_id"
            ] = violation_id
            violation[
                "evidence_image"
            ] = evidence_filename
            violation[
                "evidence_url"
            ] = evidence_url
            saved_violation_ids.append(
                violation_id
            )
        # ====================================================
        # FINAL API RESPONSE
        # ====================================================

        return {
            "success": True,
            "system":
                "Smart Helmet AI",
            "version":
                "0.6.0",
            "detected_at":
                detection_timestamp,
            "date":
                detection_date,
            "time":
                detection_clock_time,
            # ------------------------------------------------
            # DETECTION COUNTS
            # ------------------------------------------------
            "motorcycles_detected":
                len(motorcycles),
            "people_detected":
                len(people),
            "associated_riders":
                len(rider_results),
            # ------------------------------------------------
            # RIDERS
            # ------------------------------------------------
            "riders":
                rider_results,
            # ------------------------------------------------
            # HELMET RESULTS
            # ------------------------------------------------
            "helmet_detections":
                clean_helmet_results,
            # ------------------------------------------------
            # VIOLATIONS
            # ------------------------------------------------
            "violation_detected":
                len(violations) > 0,
            "violation_count":
                len(violations),
            "saved_violation_ids":
                saved_violation_ids,
            "violations":
                violations,
            # ------------------------------------------------
            # NUMBER PLATE OCR
            # ------------------------------------------------
            "plate_ocr":
                plate_ocr,
            # ------------------------------------------------
            # GPS / CAMERA INFORMATION
            # ------------------------------------------------
            "location_data": {
                "location":
                    location,
                "latitude":
                    latitude,
                "longitude":
                    longitude,
                "camera_id":
                    camera_id,
                "gps_received":
                    (
                        latitude is not None
                        and
                        longitude is not None
                    )
            },
            # ------------------------------------------------
            # EVIDENCE
            # ------------------------------------------------
            "evidence": {
                "capture_required":
                    len(violations) > 0,
                "timestamp":
                    detection_timestamp,
                "image_saved":
                    evidence_filename
                    is not None,
                "image_filename":
                    evidence_filename,
                "image_url":
                    evidence_url,
                "location":
                    location,
                "latitude":
                    latitude,
                "longitude":
                    longitude,
                "camera_id":
                    camera_id
            }
        }
    # ========================================================
    # DELETE TEMPORARY IMAGE
    # ========================================================
    finally:
        if os.path.exists(
            image_path
        ):
            os.remove(
                image_path
            )
# ============================================================
# VEHICLE SEARCH API
# ============================================================
@app.get("/vehicle-search")
def vehicle_search(
    vehicle_number: str
):
    search_value = (
        vehicle_number
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )
    if not search_value:
        raise HTTPException(
            status_code=400,
            detail="Vehicle number is required"
        )
    all_violations = (
        get_all_violations()
    )
    matches = []
    for violation in all_violations:
        matched = False
        # ----------------------------------------------------
        # CHECK CONFIRMED VEHICLE NUMBER
        # ----------------------------------------------------
        stored_number = (
            violation.get(
                "vehicle_number"
            )
        )
        if stored_number:
            clean_stored = (
                str(stored_number)
                .upper()
                .replace(" ", "")
                .replace("-", "")
            )
            if (
                search_value
                in clean_stored
            ):
                matched = True
        # ----------------------------------------------------
        # CHECK OCR CANDIDATES
        # ----------------------------------------------------
        plate_candidates = (
            violation.get(
                "plate_candidates"
            )
            or []
        )
        for candidate in (
            plate_candidates
        ):
            candidate_number = (
                candidate.get(
                    "vehicle_number"
                )
                or
                candidate.get(
                    "original_text"
                )
                or
                ""
            )
            clean_candidate = (
                str(candidate_number)
                .upper()
                .replace(" ", "")
                .replace("-", "")
            )
            if (
                search_value
                in clean_candidate
            ):
                matched = True
                break
        if matched:
            matches.append(
                violation
            )
    return {
        "success": True,
        "query": vehicle_number,
        "count": len(matches),
        "violations": matches
    }
