# Deep Learning Based Multi-Object Vehicle Detection and Tracking System

<p align="center">
  <strong>Undergraduate Graduation Design Project</strong><br>
  YOLOv8 + ByteTrack · PyQt5 GUI · Multi-Vehicle Tracking · JSON Annotation System
</p>

---

## Project Description

**Project Title**: Deep Learning Based Multi-Object Vehicle Detection and Tracking System

This project is my **undergraduate graduation design**. It is developed based on open-source projects, with extensive research, code analysis, refactoring, and feature improvements carried out independently.

---

## Key Improvements & Innovations (Compared to Original)

| Area | Original Version | This Version |
|------|-----------------|--------------|
| Detection Model | YOLOv5 | **YOLOv8** (faster, more accurate) |
| Tracking Algorithm | DeepSORT (requires extra weight files) | **ByteTrack** (built into Ultralytics, no extra files needed) |
| User Interface | No GUI | **Full PyQt5 graphical interface** |
| Vehicle Tracking | Single target | **Multi-select tracking** (3+ vehicles simultaneously) |
| Trajectory Display | Fixed thickness | **Progressive thickened trajectory** (thinner at tail, thicker at head) |
| Render Filtering | None | **Filter rendering** (non-tracked vehicles hidden when selection active) |
| Playback System | Real-time detection only | **Three-level priority**: JSON Annotation > Rendered Video > New Detection |
| Result Saving | None | JSON annotation files + annotated MP4 videos |
| Code Structure | Single file | **Modular** (`core / tracker / ui` layered architecture) |

---

## System Features

- **Real-time vehicle detection and tracking** — supports car, truck, bus, motorcycle, bicycle, person
- **Click-to-track** — click directly on detection boxes in the video, or click items in the right-side vehicle list to toggle tracking
- **Multi-vehicle tracking** — track multiple vehicles simultaneously; tracked targets are highlighted with a white border, others are hidden
- **Progressive trajectory** — vehicle trails accumulate frame by frame in real time, with gradient line thickness showing direction of travel
- **Playback control** — draggable progress bar, pause / resume, loop playback
- **Video source switching** — supports webcam (index 0) and local video files (mp4 / avi / mov / mkv / wmv / flv / ts)
- **Three-level playback priority** — JSON annotation replay (with filtering) → pre-rendered video → fresh AI detection
- **Auto-save results** — JSON annotation files and annotated MP4 videos saved automatically; no re-detection needed on next open
- **Real-time system log** — right-side log panel shows running status, FPS, vehicle count, and more
- **Re-detect button** — clears previous results and forces a fresh AI detection run

---

## Project Structure

```
Vehicle-tracking-main/
│
├── main.py                            # Entry point — run this file to start the system
│
├── core/                              # Configuration & utility layer
│   ├── __init__.py
│   ├── constants.py                   # Vehicle color mapping, default paths
│   ├── utils.py                       # Utility functions (path, color conversion, etc.)
│   ├── config.py                      # OPT config class (reads YAML, initializes parameters)
│   └── model_loader.py                # ModelLoadThread (background YOLOv8 loader)
│
├── tracker/                           # Detection & tracking logic layer
│   ├── __init__.py
│   ├── trail_manager.py               # Trail management (rolling window + full accumulation + progressive drawing)
│   ├── annotation_io.py               # JSON annotation file read/write
│   ├── detect_mode.py                 # ByteTrack detection main loop
│   └── tracking_thread.py             # TrackingThread (dispatches three playback modes)
│
├── ui/                                # GUI layer
│   ├── __init__.py
│   ├── styles.py                      # QSS style strings (login + main window)
│   ├── video_label.py                 # ClickableVideoLabel (clickable video widget)
│   ├── login_window.py                # Login window
│   └── main_window.py                 # Main window (video display, controls, log)
│
├── videos/                            # Test video directory — place test videos here
│   └── *.avi                           # Supported:  avi
│
├── application/
│   └── main/
│       └── inference/
│           └── output/                # Output directory — all results saved here
│               ├── *_annotations.json          # Per-frame detection result annotations
│               └── *_out_annotated.mp4         # Annotated output video with boxes & trails
│
├── models/                            # Model weight directory
│   └── yolov8n.pt                     # YOLOv8 weights (auto-downloaded on first run)
│
├── settings/
│   └── config.yml                     # System configuration file
│
├── yolo_train/                        # YOLOv8 training scripts
│
├── requirements.txt                   # Python dependency list
└── README.md                          # This file
```

---

## Test Data & Output Results

### Test Videos

Place test video files in:
```
.\videos\
```

Supported formats: `.mp4`  `.avi`  `.mov`  `.mkv`  `.wmv`  `.flv`  `.ts`

You can also click **"Select video file..."** in the interface to load a video from any location.

---

### Output Results

All detection results are automatically saved to:
```
.\application\main\inference\output\
```

| File | Description |
|------|-------------|
| `videoname_annotations.json` | Full per-frame annotation data including bounding boxes, global IDs, and trajectory points |
| `videoname_out_annotated.mp4` | Complete annotated video with detection boxes, ID labels, and trajectory lines |

> **Note**: The JSON annotation file enables instant replay and multi-select filter tracking the next time the same video is opened. It is recommended to keep this file. To force re-detection, click the **↺ Re-detect** button.

---

## Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.8 – 3.10 recommended |
| CUDA (optional) | GPU acceleration supported; automatically falls back to CPU if unavailable |
| Operating System | Windows 10 / 11 (primary test environment) |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/xiaoming123-xm/xiaoming123-xm-Deep-Learning-Based-Multi-Object-Vehicle-Detection-System.git
cd vehicle-tracking
```

### 2. Install Dependencies

```bash
# Recommended: use a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

> If you need GPU acceleration, install the appropriate CUDA version of PyTorch from the [PyTorch official website](https://pytorch.org/) before running `pip install -r requirements.txt`.

### 3. Prepare Test Videos

Place video files in the `videos/` directory:

```
videos/
├── test1.avi
└── test2.avi
```

### 4. Run the System

```bash
python main.py
```

A login window will appear. Use the following credentials:

| Username | Password |
|----------|----------|
| `abcde`  | `123456` |

### 5. CLI Mode (Optional)

```bash
python main.py cli
```

---

## How to Use

```
┌──────────────────────────────────────────────────────────────┐
│  Login Window  →  Main Window                                │
│                                                              │
│  1. Select video source: webcam (0) or local video file      │
│  2. Click [ ▶ Start Detection / Annotate ]                   │
│     - JSON annotation exists  →  instant replay             │
│       (supports multi-select filter tracking)                │
│     - Pre-rendered video exists  →  direct playback          │
│     - No previous results  →  AI detection + auto-save       │
│  3. Click a detection box in the video                       │
│     OR click an item in the right-side vehicle list          │
│     →  toggle tracking for that vehicle                      │
│  4. Click again  OR  click [ ✕ Clear All Tracking ]          │
│     →  deselect and show all vehicles                        │
│  5. Drag the progress bar to jump to any frame               │
│  6. Click [ ↺ Re-detect ] to delete old results              │
│     and run a fresh AI detection pass                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Dependencies

`requirements.txt`:

```
ultralytics>=8.0.0
torch>=1.10.0
torchvision>=0.11.0
opencv-python>=4.5.0
PyQt5>=5.15.0
numpy>=1.21.0
PyYAML>=5.4.0
```

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

The configuration file is located at `settings/config.yml`. It will be auto-generated on first run if it does not exist.

```yaml
device: cuda          # Device to use: cuda (GPU) or cpu
classes:              # Target class IDs to detect (COCO dataset)
  - 0                 # person
  - 1                 # bicycle
  - 2                 # car
  - 3                 # motorcycle
  - 5                 # bus
  - 7                 # truck
conf_thres: 0.25      # Confidence threshold (lower = more detections)
iou_thres: 0.5        # IOU threshold for NMS duplicate suppression
```

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      main.py  (Entry Point)                │
├─────────────────┬──────────────────────┬───────────────────┤
│     core/       │      tracker/        │       ui/         │
│  Config & Utils │  Detection & Track   │  Graphical UI     │
│                 │                      │                   │
│  constants      │  trail_manager       │  styles           │
│  utils          │  annotation_io       │  video_label      │
│  config (OPT)   │  detect_mode         │  login_window     │
│  model_loader   │  tracking_thread     │  main_window      │
└────────────────────────────────────────────────────────────┘
          │                   │
   YOLOv8 (Ultralytics) + ByteTrack (built-in)
          │                   │
   OpenCV frame processing   PyQt5 GUI rendering
```

| Module | Technology | Description |
|--------|-----------|-------------|
| Object Detection | Ultralytics YOLOv8 | Real-time multi-class vehicle detection |
| Object Tracking | ByteTrack (built-in) | Stable cross-frame IDs, no extra weights needed |
| GUI Framework | PyQt5 | Multi-threaded GUI; video frames passed via signal-slot |
| Video Processing | OpenCV | Frame reading, annotation drawing, video writing |
| Data Persistence | JSON | Annotation storage for fast replay and filtering |
| Configuration | PyYAML | Flexible parameter configuration |

---

## FAQ

**Q: Why is the first run slow?**  
A: On the first run, the YOLOv8 weight file (`yolov8n.pt`, ~6 MB) is downloaded automatically and saved to the `models/` directory. Subsequent runs load the local file and start immediately.

**Q: Can it run without a GPU?**  
A: Yes. The system automatically detects whether CUDA is available and falls back to CPU if not. Detection will be slower on CPU but all features remain fully functional.

**Q: How do I speed up detection?**  
A: Increase `conf_thres` in `settings/config.yml` to reduce low-confidence detections, or run on a machine with a CUDA-compatible GPU.

**Q: Why does the same video open instantly the second time?**  
A: After the first detection pass, results are saved as a JSON annotation file. On subsequent opens, the system loads the annotation directly and replays it without re-running AI detection. Click **↺ Re-detect** to force a fresh detection if needed.

**Q: Where are the output files saved?**  
A: All output files are saved to `application\main\inference\output\`, including `_annotations.json` annotation files and `_out_annotated.mp4` annotated videos.

---

## License

This project is developed for academic research and undergraduate graduation purposes only.  
Not intended for commercial use.

---

<p align="center">
  Made with ❤️ as an Undergraduate Graduation Design
</p>