"""
Level 6 — Video Object Tracking and Counting
Runs YOLOv8 with built-in object tracking across video frames,
counting unique objects per class rather than re-detecting the
same object every frame.
"""

from ultralytics import YOLO
import cv2
from collections import defaultdict

def main():
    model = YOLO("yolov8n.pt")

    # Change this to test a different clip - output filename follows automatically
    video_path = "traffic5.mp4"
    output_path = video_path.replace("traffic", "output_tracked")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicle_class_ids = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    unique_ids_by_class = defaultdict(set)
    peak_in_frame_by_class = defaultdict(int)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Lowered conf from the 0.25 default since smaller/rarer classes
        # like motorcycle tend to score lower confidence than cars
        results = model.track(frame, persist=True, verbose=False, conf=0.15)
        result = results[0]

        current_counts = defaultdict(int)

        if result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                class_id = int(box.cls[0])
                if class_id not in vehicle_class_ids:
                    continue

                label = vehicle_class_ids[class_id]
                obj_id = int(track_id)
                unique_ids_by_class[label].add(obj_id)
                current_counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} #{obj_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for label, count in current_counts.items():
            if count > peak_in_frame_by_class[label]:
                peak_in_frame_by_class[label] = count

        total_in_frame = sum(current_counts.values())
        total_unique = sum(len(ids) for ids in unique_ids_by_class.values())

        cv2.putText(frame, f"In frame: {total_in_frame}  |  Unique total: {total_unique}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Vehicle Tracking", frame)

        if frame_count % 30 == 0:
            breakdown = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in current_counts.items())
            print(f"Frame {frame_count}: {total_in_frame} in frame ({breakdown}), {total_unique} unique so far")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n{video_path} -> Processed {frame_count} frames")
    print(f"\nUnique objects tracked, by class:")
    total_unique = 0
    for label in vehicle_class_ids.values():
        count = len(unique_ids_by_class[label])
        peak = peak_in_frame_by_class[label]
        total_unique += count
        print(f"  {label}: {count} unique  (peak {peak} simultaneously in frame)")
    print(f"  Total: {total_unique} unique objects")
    print(f"\nSaved annotated video as {output_path}")


if __name__ == "__main__":
    main()