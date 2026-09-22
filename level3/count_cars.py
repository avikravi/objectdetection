"""
Vehicle Detection and Counting - Proof of Concept
Uses YOLOv8 to detect and count cars in a traffic image.
"""

from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n.pt")

    image_path = "traffic.jpg"
    results = model(image_path)
    result = results[0]

    # COCO class IDs: 2 = car, 5 = bus, 7 = truck
    vehicle_class_ids = {2: "car", 5: "bus", 7: "truck"}

    img = cv2.imread(image_path)
    counts = {"car": 0, "bus": 0, "truck": 0}

    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id in vehicle_class_ids:
            label = vehicle_class_ids[class_id]
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    total = sum(counts.values())
    print(f"Total vehicles detected: {total}")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    cv2.imwrite("output.jpg", img)
    print("Saved annotated image as output.jpg")

    cv2.imshow("Vehicle Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()