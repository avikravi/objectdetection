"""
Level 6 — Grounding DINO (Transformer-based Open-Vocabulary Detection)
Adds: per-class colors, cross-class duplicate suppression, and basic
label-overlap avoidance for readability.
"""

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import cv2
import numpy as np

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def main():
    # Change this to test a different image - output filename follows automatically
    image_path = "warehouse4.jpg"
    output_path = image_path.replace("warehouse", "output")

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")

    class_labels = ["helmet", "person", "safety vest",
                     "electrical cabinet", "hand pallet truck",
                     "cardboard box", "wooden crate"]
    text_query = ". ".join(class_labels) + "."

    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Lowered from 0.3 to catch smaller/partially occluded objects
    # like the pallet truck, at the cost of slightly more noise
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        box_threshold=0.22,
        text_threshold=0.2,
        target_sizes=[image.size[::-1]],
    )[0]

    # Cross-class duplicate suppression: different text queries can
    # independently match the same physical object. Sort by confidence,
    # then drop any lower-confidence box that heavily overlaps a
    # higher-confidence one already kept, regardless of label.
    detections = list(zip(
        results["boxes"].tolist(), results["scores"].tolist(), results["labels"]
    ))
    detections.sort(key=lambda d: d[1], reverse=True)

    kept = []
    for box, score, label in detections:
        is_duplicate = any(compute_iou(box, k[0]) > 0.6 for k in kept)
        if not is_duplicate:
            kept.append((box, score, label.strip()))

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    scale_factor = max(width, height) / 1000
    box_thickness = max(2, int(3 * scale_factor))
    font_scale = max(0.45, 0.55 * scale_factor)   # reduced from 0.8
    font_thickness = max(1, int(1.5 * scale_factor))

    colors = {
        "person": (0, 255, 0),
        "helmet": (0, 165, 255),
        "safety vest": (0, 140, 255),
        "electrical cabinet": (200, 0, 200),
        "hand pallet truck": (0, 0, 255),
        "cardboard box": (255, 200, 0),
        "wooden crate": (19, 69, 139),
    }

    placed_label_boxes = []  # tracks already-drawn label rectangles

    def find_label_position(x1, y1, text_w, text_h):
        # start just above the box, drop down in steps if it collides
        # with a previously placed label
        y = y1 - 6
        for _ in range(10):
            candidate = (x1, y - text_h - 6, x1 + text_w + 8, y + 6)
            collision = any(
                not (candidate[2] < p[0] or candidate[0] > p[2] or
                     candidate[3] < p[1] or candidate[1] > p[3])
                for p in placed_label_boxes
            )
            if not collision:
                placed_label_boxes.append(candidate)
                return y
            y += text_h + 14
        placed_label_boxes.append(candidate)
        return y

    counts = {}

    for box, score, label in kept:
        counts[label] = counts.get(label, 0) + 1
        color = colors.get(label, (0, 255, 0))

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        text = f"{label} {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        label_y = find_label_position(x1, y1, text_w, text_h)
        cv2.rectangle(img, (x1, label_y - text_h - 6), (x1 + text_w + 8, label_y + 6), color, -1)
        cv2.putText(img, text, (x1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    total = sum(counts.values())
    print(f"{image_path} -> Total objects detected: {total}")
    for label, count in counts.items():
        print(f"  {label}: {count}")

    cv2.imwrite(output_path, img)
    print(f"Saved annotated image as {output_path}")


if __name__ == "__main__":
    main()