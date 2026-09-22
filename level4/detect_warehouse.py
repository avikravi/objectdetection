"""
Level 4 — Manufacturing/Logistics Scene Detection
Runs YOLOv8 on a warehouse image, reporting whatever COCO classes
actually appear (realistically: person, truck).
"""

from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")

    image_path = "warehouse.jpg"
    results = model(image_path)
    result = results[0]

    img = cv2.imread(image_path)
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

    total = sum(counts.values())
    print(f"Total objects detected: {total}")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    if not counts:
        print("  No COCO classes detected in this scene.")

    cv2.imwrite("output.jpg", img)
    print("Saved annotated image as output.jpg")

    cv2.imshow("Warehouse Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()