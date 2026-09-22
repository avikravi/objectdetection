"""
Level 5 — Open-Vocabulary Detection (YOLO-World)
Detects custom, user-defined object classes with no training required —
unlike Levels 3/4, which are limited to COCO's fixed 80 classes.
"""

from ultralytics import YOLOWorld
import cv2

def main():
    model = YOLOWorld("yolov8s-world.pt")  # downloads automatically on first run

    custom_classes = ["helmet", "person", "safety vest",
                       "electrical cabinet", "pallet jack",
                       "tablet", "warning sign"]
    model.set_classes(custom_classes)

    image_path = "warehouse4.jpg"
    output_path = image_path.replace("warehouse", "output")

    # Lowered conf slightly to recover electrical cabinet matches,
    # which shifted below 0.08 at the higher imgsz resolution
    results = model.predict(image_path, conf=0.06, iou=0.5, imgsz=1280)
    result = results[0]

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    scale_factor = max(width, height) / 1000
    box_thickness = max(2, int(3 * scale_factor))
    font_scale = max(0.6, 0.8 * scale_factor)
    font_thickness = max(1, int(2 * scale_factor))

    colors = {
        "person": (0, 255, 0),
        "helmet": (0, 165, 255),
        "safety vest": (0, 140, 255),
        "electrical cabinet": (200, 0, 200),
        "pallet jack": (0, 0, 255),
        "tablet": (255, 0, 0),
        "warning sign": (0, 255, 255),
    }

    counts = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = custom_classes[class_id]
        confidence = float(box.conf[0])
        counts[label] = counts.get(label, 0) + 1
        color = colors.get(label, (0, 255, 0))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        text = f"{label} {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        cv2.rectangle(img, (x1, y1 - text_h - 12), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(img, text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    total = sum(counts.values())
    print(f"{image_path} -> Total objects detected: {total}")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    if not counts:
        print("  No objects detected - try lowering conf further or rephrasing classes.")

    cv2.imwrite(output_path, img)
    print(f"Saved annotated image as {output_path}")

    cv2.imshow("Open-Vocabulary Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()