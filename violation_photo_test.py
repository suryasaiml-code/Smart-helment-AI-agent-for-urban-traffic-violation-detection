import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Input photo
image_path = "bike.jpg"

# Read image
frame = cv2.imread(image_path)

if frame is None:
    print("ERROR: bike.jpg not found.")
    exit()

# Detect objects
results = model(frame, verbose=False)

motorcycles = 0
people = 0

# Check detections
for box in results[0].boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    if confidence < 0.40:
        continue

    # COCO:
    # 0 = person
    # 3 = motorcycle

    if class_id == 0:
        people += 1

    elif class_id == 3:
        motorcycles += 1

# Draw detections
annotated = results[0].plot()

# Display counts
cv2.putText(
    annotated,
    f"Motorcycles: {motorcycles}",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    annotated,
    f"People: {people}",
    (30, 75),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

print("================================")
print(" SMART HELMET PHOTO TEST")
print("================================")
print("Motorcycles:", motorcycles)
print("People:", people)

# Save result
output = "violation_photo_result.jpg"

cv2.imwrite(output, annotated)

print("Result saved as:", output)

# Show image
cv2.imshow(
    "Smart Helmet - Photo Test",
    annotated
)

cv2.waitKey(0)
cv2.destroyAllWindows()