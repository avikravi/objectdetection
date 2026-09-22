"""
Level 2 — HOG + SVM Pedestrian Detection (Historical Bridge)
"""

import cv2
import numpy as np

def non_max_suppression(boxes, overlap_thresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(y2)

    keep = []
    while len(order) > 0:
        i = order[-1]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[order[:-1]]
        order = np.delete(order, np.concatenate(([len(order) - 1], np.where(overlap > overlap_thresh)[0])))

    return boxes[keep].astype(int)


def main():
    # Change this to test a different image - output filename follows automatically
    image_path = "people4.jpg"
    output_path = image_path.replace("people", "output")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    boxes, weights = hog.detectMultiScale(
        img, winStride=(8, 8), padding=(8, 8), scale=1.05
    )

    # Non-max suppression: collapses the many overlapping duplicate
    # boxes HOG+SVM produces per person into one clean box each
    boxes = non_max_suppression(np.array(boxes), overlap_thresh=0.4)

    print(f"{image_path} -> Detected {len(boxes)} pedestrian(s) after NMS")

    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "person", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"Saved annotated image as {output_path}")

    cv2.imshow("HOG+SVM Pedestrian Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()