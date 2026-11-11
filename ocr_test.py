import cv2
from paddleocr import PaddleOCR

IMAGE_PATH = "bike.jpg"

# Read image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: bike.jpg not found.")
    exit()

print("\n==============================")
print(" SMART HELMET AI - OCR TEST")
print("==============================")

# Create OCR
ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

# Run OCR
result = ocr.ocr(IMAGE_PATH, cls=False)

print("\nDetected text:")

found = False

for line in result:

    if line is None:
        continue

    for item in line:

        text = item[1][0]
        confidence = item[1][1]

        if confidence >= 0.50:

            found = True

            print(
                f"  {text} "
                f"- {confidence * 100:.2f}%"
            )

if not found:
    print("  No readable text detected.")

print("\n==============================")
print(" OCR TEST FINISHED")
print("==============================")