import os
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

image_path = "bike.jpg"

result = CLIENT.infer(
    image_path,
    model_id="motor-cycle-helmet/6"
)

image = cv2.imread(image_path)

for p in result["predictions"]:

    x = int(p["x"])
    y = int(p["y"])
    w = int(p["width"])
    h = int(p["height"])

    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x + w // 2
    y2 = y + h // 2

    label = f"{p['class']} {p['confidence'] * 100:.1f}%"

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

output = "helmet_detection_result.jpg"

cv2.imwrite(output, image)

print("Detection completed!")
print("Saved:", output)