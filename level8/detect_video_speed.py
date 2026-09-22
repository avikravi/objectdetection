"""
Level 8 — Video Tracking with Speed Estimation and Data Export
Estimates each tracked vehicle's speed, flags vehicles over a threshold,
and extracts a structured summary table (printed + saved as CSV) of
every vehicle's speed data across the full video.

IMPORTANT: speed is only as accurate as the calibration below. Without
a true calibrated reference (e.g. surveyed lane markings, known camera
geometry), this is an approximation - stated explicitly in the output.
"""

from ultralytics import YOLO
import cv2
import csv
from collections import defaultdict, deque

def main():
    model = YOLO("yolov8n.pt")

    # Change this to test a different clip - output filenames follow automatically
    video_path = "traffic4.mp4"
    output_path = video_path.replace("traffic", "output_speed")
    csv_path = video_path.replace("traffic", "speed_data").replace(".mp4", ".csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # --- CALIBRATION - adjust this for your specific video ---
    PIXELS_PER_METER = 32.0
    MPH_CONVERSION = 2.23694
    SPEED_THRESHOLD_MPH = 10

    vehicle_class_ids = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    track_history = defaultdict(lambda: deque(maxlen=5))

    # Full speed record per vehicle ID, kept for the whole video -
    # this is the extraction pipeline: every mph reading, per vehicle,
    # per frame, collected for later summary
    vehicle_data = defaultdict(lambda: {"class": None, "speeds": [], "first_frame": None, "last_frame": None})

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model.track(frame, persist=True, verbose=False, conf=0.15)
        result = results[0]

        if result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                class_id = int(box.cls[0])
                if class_id not in vehicle_class_ids:
                    continue

                label = vehicle_class_ids[class_id]
                obj_id = int(track_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                track_history[obj_id].append((cx, cy, frame_count))

                speed_text = ""
                mph = 0
                if len(track_history[obj_id]) >= 2:
                    (x_old, y_old, f_old) = track_history[obj_id][0]
                    (x_new, y_new, f_new) = track_history[obj_id][-1]
                    frame_gap = f_new - f_old

                    if frame_gap > 0:
                        pixel_dist = ((x_new - x_old) ** 2 + (y_new - y_old) ** 2) ** 0.5
                        seconds = frame_gap / fps
                        meters_per_sec = (pixel_dist / PIXELS_PER_METER) / seconds
                        mph = meters_per_sec * MPH_CONVERSION
                        speed_text = f" {mph:.0f}mph"

                        # Record this reading in the vehicle's data history
                        record = vehicle_data[obj_id]
                        record["class"] = label
                        record["speeds"].append(mph)
                        if record["first_frame"] is None:
                            record["first_frame"] = frame_count
                        record["last_frame"] = frame_count

                box_color = (0, 0, 255) if mph > SPEED_THRESHOLD_MPH else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"{label} #{obj_id}{speed_text}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        cv2.putText(frame, "Speed is approximate (calibrated estimate)",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(frame)
        cv2.imshow("Speed Estimation", frame)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count} processed")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- Build the summary table ---
    summary_rows = []
    for obj_id, data in vehicle_data.items():
        if not data["speeds"]:
            continue
        avg_speed = sum(data["speeds"]) / len(data["speeds"])
        max_speed = max(data["speeds"])
        min_speed = min(data["speeds"])
        duration_frames = data["last_frame"] - data["first_frame"] + 1
        summary_rows.append({
            "vehicle_id": obj_id,
            "class": data["class"],
            "avg_mph": round(avg_speed, 1),
            "max_mph": round(max_speed, 1),
            "min_mph": round(min_speed, 1),
            "frames_tracked": duration_frames,
            "over_threshold": "YES" if max_speed > SPEED_THRESHOLD_MPH else "no",
        })

    summary_rows.sort(key=lambda r: r["vehicle_id"])

    # Print formatted table to terminal
    print(f"\n{'='*80}")
    print(f"VEHICLE SPEED SUMMARY — {video_path}")
    print(f"{'='*80}")
    print(f"{'ID':<6}{'Class':<12}{'Avg MPH':<10}{'Max MPH':<10}{'Min MPH':<10}{'Frames':<10}{'Over Limit':<10}")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['vehicle_id']:<6}{row['class']:<12}{row['avg_mph']:<10}{row['max_mph']:<10}"
              f"{row['min_mph']:<10}{row['frames_tracked']:<10}{row['over_threshold']:<10}")
    print("-" * 80)
    print(f"Total vehicles tracked: {len(summary_rows)}")
    over_count = sum(1 for r in summary_rows if r["over_threshold"] == "YES")
    print(f"Vehicles exceeding {SPEED_THRESHOLD_MPH}mph threshold: {over_count}")

    # Save as CSV - a real extractable data artifact
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vehicle_id", "class", "avg_mph", "max_mph",
                                                  "min_mph", "frames_tracked", "over_threshold"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved annotated video as {output_path}")
    print(f"Saved speed data table as {csv_path}")
    print(f"\nNote: speeds are estimates based on PIXELS_PER_METER = {PIXELS_PER_METER}")
    print("Calibrate this value against a known real-world distance in your specific footage for accuracy.")


if __name__ == "__main__":
    main()