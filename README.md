# Object Detection Pipeline

Eight progressively more advanced object detection projects, from a basic
Haar Cascade face detector up to video tracking with speed estimation.

## Setup

```bash
git clone https://github.com/avikravi/objectdetection.git
cd objectdetection
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Every script is run **from inside its own level folder** (paths are relative),
and opens a live preview window — press `q` to quit a video early.

## Levels

**Level 1 — Face detection**
```bash
cd level1 && python face.py
```
Auto-downloads a sample image on first run. No setup needed.

**Level 2 — Pedestrian detection (HOG+SVM)**
```bash
cd level2 && python detect_pedestrians.py
```
Runs on `people4.jpg` by default. To try another sample image, edit the
`image_path` variable near the top of the script.

**Level 3 — YOLO web app (Flask)**
```bash
cd level3 && python app.py
```
Open `http://127.0.0.1:5000`, upload any image, see detections.

**Level 4 — Warehouse/logistics detection**
```bash
cd level4 && python detect_warehouse.py
```
Runs on `warehouse.jpg` by default.

**Level 5 — Open-vocabulary detection (YOLO-World)**
```bash
cd level5 && python detect_openvocab.py
```
Detects custom classes (helmet, safety vest, etc.) with no training. Edit
`image_path` to try `warehouse1–4.jpg`.

**Level 6 — Grounding DINO (transformer-based open-vocab)**
```bash
cd level6 && python detect_grounding_dino.py
```
First run downloads the model from Hugging Face — needs internet access once.

**Level 7 — Video tracking**
```bash
cd level7 && python detect_video_tracking.py
```
Video files aren't included in this repo (too large for GitHub). Drop your
own `.mp4` into `level7/`, then edit the `video_path` variable at the top of
the script to match your filename.

**Level 8 — Video speed estimation**
```bash
cd level8 && python detect_video_speed.py
```
Same as Level 7 — supply your own video and update `video_path`. Also
outputs a CSV of per-vehicle speed data. Speeds are estimates: adjust the
`PIXELS_PER_METER` constant near the top of the script to calibrate against
your footage for real accuracy.

## Notes

- All input paths are hardcoded near the top of each script — just edit the
  variable to point at a different file.
- YOLO model weights (`yolov8n.pt`, `yolov8s-world.pt`) are already included
  in the repo, so no separate download is needed for Levels 1–5, 7, 8.
