import cv2
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# ==========================================
# SMART HELMET AI - IMAGE TEST
# ==========================================

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not found.")
    exit()

IMAGE_PATH = "bike.jpg"

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

HELMET_MODEL = "motor-cycle-helmet/6"

# ==========================================
# READ IMAGE
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: bike.jpg not found.")
    exit()

print()
print("===================================")
print(" SMART HELMET AI - IMAGE TEST")
print("===================================")

# ==========================================
# ROBoflow HELMET DETECTION
# ==========================================

print("\nRunning helmet detection...")

result = CLIENT.infer(
    IMAGE_PATH,
    model_id=HELMET_MODEL
)

predictions = result.get("predictions", [])

helmet_count = 0
no_helmet_count = 0

# ==========================================
# DRAW DETECTIONS
# ==========================================

for prediction in predictions:

    class_name = prediction["class"]
    confidence = float(prediction["confidence"])

    if class_name.lower() not in [
        "helmet",
        "withouthelmet"
    ]:
        continue

    x = int(prediction["x"])
    y = int(prediction["y"])
    w = int(prediction["width"])
    h = int(prediction["height"])

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    if class_name.lower() == "helmet":

        helmet_count += 1
        color = (0, 255, 0)
        label = "HELMET"

    else:

        no_helmet_count += 1
        color = (0, 0, 255)
        label = "NO HELMET"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        image,
        f"{label} {confidence * 100:.1f}%",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

# ==========================================
# SUMMARY
# ==========================================

print()
print("Helmet detections :", helmet_count)
print("No-helmet detections :", no_helmet_count)

if no_helmet_count > 0:

    print()
    print("🚨 VIOLATION DETECTED")
    print("Type: NO HELMET")

else:

    print()
    print("✅ No helmet violation detected")

# ==========================================
# SAVE RESULT
# ==========================================

output = "test_full_detection_result.jpg"

cv2.imwrite(
    output,
    image
)

print()
print("Result saved as:")
print(output)

# ==========================================
# SHOW RESULT
# ==========================================

cv2.imshow(
    "Smart Helmet AI - Test Result",
    image
)

print()
print("Press Q to close the image.")

while True:

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()