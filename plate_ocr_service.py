import torch
from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
import tempfile
import os
import re

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="Smart Helmet AI - Plate OCR Service",
    version="1.0.0"
)

# ==========================================
# LOAD PADDLE OCR
# ==========================================

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    show_log=False,
    enable_mkldnn=False
)

print("PaddleOCR loaded successfully.")

# ==========================================
# HELPER FUNCTION
# ==========================================

def clean_text(text):
    """
    Remove spaces and symbols so that:
    TN-02-AV-6447
    becomes:
    TN02AV6447
    """

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


def looks_like_vehicle_number(text):
    """
    Basic filter for likely Indian vehicle
    registration numbers.
    """

    cleaned = clean_text(text)

    # Avoid very short/very long OCR results
    if len(cleaned) < 7 or len(cleaned) > 12:
        return False

    # Must contain letters AND numbers
    has_letter = any(
        char.isalpha()
        for char in cleaned
    )

    has_number = any(
        char.isdigit()
        for char in cleaned
    )

    return has_letter and has_number


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "status": "online",
        "service": "Smart Helmet AI Plate OCR",
        "message": "OCR service is running"
    }


# ==========================================
# OCR ENDPOINT
# ==========================================

@app.post("/ocr")
async def detect_plate(
    file: UploadFile = File(...)
):

    contents = await file.read()

    suffix = os.path.splitext(
        file.filename or ""
    )[1]

    if not suffix:
        suffix = ".jpg"

    # ======================================
    # SAVE TEMPORARY IMAGE
    # ======================================

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(contents)
        image_path = temp.name

    try:

        # ==================================
        # RUN OCR
        # ==================================

        result = ocr.ocr(
            image_path,
            cls=False
        )

        detected_text = []

        plate_candidates = []

        # ==================================
        # PROCESS OCR RESULT
        # ==================================

        if result:

            for page in result:

                if not page:
                    continue

                for line in page:

                    if (
                        not line
                        or len(line) < 2
                    ):
                        continue

                    text = str(
                        line[1][0]
                    ).strip()

                    confidence = float(
                        line[1][1]
                    )

                    item = {
                        "text": text,
                        "confidence": round(
                            confidence,
                            3
                        )
                    }

                    detected_text.append(
                        item
                    )

                    # Ignore weak OCR
                    if confidence < 0.60:
                        continue

                    # Check if it looks
                    # like a vehicle number
                    if looks_like_vehicle_number(
                        text
                    ):

                        plate_candidates.append({
                            "original_text":
                                text,

                            "vehicle_number":
                                clean_text(text),

                            "confidence":
                                round(
                                    confidence,
                                    3
                                )
                        })

        # ==================================
        # SELECT BEST PLATE
        # ==================================

        best_plate = None

        if plate_candidates:

            plate_candidates.sort(
                key=lambda x:
                    x["confidence"],
                reverse=True
            )

            best_plate = (
                plate_candidates[0]
            )

        # ==================================
        # RETURN RESULT
        # ==================================

        return {
            "success": True,

            "vehicle_number":
                (
                    best_plate[
                        "vehicle_number"
                    ]
                    if best_plate
                    else None
                ),

            "plate_confidence":
                (
                    best_plate[
                        "confidence"
                    ]
                    if best_plate
                    else 0
                ),

            "plate_candidates":
                plate_candidates,

            "all_detected_text":
                detected_text
        }

    except Exception as error:

        return {
            "success": False,
            "vehicle_number": None,
            "error": str(error)
        }

    finally:

        if os.path.exists(
            image_path
        ):
            os.remove(
                image_path
            )