"""
Simple Flask web app for YOLO object detection.
Upload an image on the left, see detection results on the right.
"""

from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

vehicle_class_ids = {2: "car", 5: "bus", 7: "truck"}


@app.route("/", methods=["GET", "POST"])
def index():
    original_image = None
    result_image = None
    detection_summary = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            # Save the uploaded original
            original_path = os.path.join(UPLOAD_FOLDER, "original.jpg")
            file.save(original_path)
            original_image = original_path

            # Run detection
            img = cv2.imread(original_path)
            results = model(img, verbose=False)
            result = results[0]

            counts = {}
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])
                counts[label] = counts.get(label, 0) + 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            result_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
            cv2.imwrite(result_path, img)
            result_image = result_path

            detection_summary = counts if counts else {"No objects detected": 0}

    return render_template(
        "index.html",
        original_image=original_image,
        result_image=result_image,
        detection_summary=detection_summary,
    )


if __name__ == "__main__":
    app.run(debug=True)