# InnerveX-Final

InnerveX-Final is a multispectral fusion and detection toolkit that combines visible imagery (Pi Camera IMX sensors) with MLX90640 thermal data and YOLO object detection. It provides multiple Tkinter GUI apps for live visualization, thermal-visual alignment and fusion, live YOLO inference, detection logging, heatmaps, and a small reporting pipeline to generate an HTML intelligence report.

---
Table of contents
- Project overview
- Key features
- Repo layout
- Requirements
  - Hardware
  - System / OS
  - Python packages
- Quick install (Raspberry Pi recommended)
- Configuration
  - YOLO weights
  - Stereo homography YAML
  - MLX alignment JSON
- Running the apps
  - apps/spectra_main.py
  - apps/spectra_gui_2.py
  - apps/multispectral_fusion_app_gui.py
  - detect4c_gui.py
- Detection logging & reporting
  - parser.py
  - data/report.html
- Testing and utilities
- Troubleshooting
- Development notes / code structure
- Contributing
- License

---

Project overview
This repository is intended for live multispectral surveillance and experimentation:
- Capture one or two visible camera feeds (IMX519 / IMX500 are referenced)
- Read MLX90640 thermal frames over I2C
- Align/warp thermal images into the visible frame using a homography plus manual translate/scale
- Fuse thermal and visible imagery with configurable alpha weights
- Run YOLOv8 object detection on visible frames and annotate all views
- Log detections to CSV and create a small HTML report summarizing detections over time

Key features
- Multiple Tkinter-based GUIs for dashboards and monitoring (3-view, fused view, interactive alignment)
- Thermal -> visible registration and per-pixel fusion (configurable alpha)
- YOLOv8 integration (Ultralytics) for realtime detection
- Detection logging to data/detections.csv
- Reporting: parser.py turns CSV into an HTML SPECTRA report and output assets
- Small helper scripts for camera testing and heatmap visualisation

Repository layout (high-level)
- apps/
  - spectra_main.py — full featured dashboard (IMX500/IMX519 + MLX90640 + YOLO + dashboard)
  - spectra_gui_2.py — simplified 3-view app and demo
  - multispectral_fusion_app_gui.py — another app variant focusing on fusion
- detect4c_gui.py — standalone multispectral dashboard, fusion & MLX handling
- heatmap.py — utilities for thermal heatmaps
- parser.py — CSV ? HTML report builder (data/report.html)
- data/ — stores stereo_alignment_matrix.yaml, mlx_manual_align.json, detections.csv, report.html
- models/ or repo root — expected location for YOLO weights (yolov8n.pt) or similar
- test/ — small test scripts (e.g., test/opencv_bothcam.py)
- LICENSE — MIT

Requirements

Hardware
- Raspberry Pi (or Linux machine with Picamera2-compatible cameras)
- One or two Pi cameras (IMX519/IMX500 referenced in code)
- MLX90640 thermal sensor (I2C)
- Adequate CPU/GPU for YOLO inference (yolov8n is small; CPU inference possible but slower)

System / OS
- Raspberry Pi OS (Bullseye/Bookworm) or another Linux distro with libcamera/Picamera2 support
- I2C enabled for MLX90640

Python packages
The project uses the following Python packages (non-exhaustive):
- ultralytics (YOLOv8)
- numpy
- opencv-python (or opencv-python-headless for headless installs)
- pillow
- pyyaml
- matplotlib (for embedded heatmaps)
- picamera2 (for Pi camera capture)
- adafruit-circuitpython-mlx90640
- adafruit-blinka / board / busio (for I2C on Raspberry Pi)

Suggested requirements.txt (example)
Create a requirements.txt in the project root with the following (version pinning recommended for production):

numpy
opencv-python
pillow
pyyaml
matplotlib
ultralytics
picamera2
adafruit-circuitpython-mlx90640

Note: On Raspberry Pi, Picamera2 and some hardware-related packages may be installed through apt or via the Raspberry Pi OS package ecosystem. Follow Pi-specific install instructions below.

Quick install (Raspberry Pi recommended)
1. Clone
   - git clone https://github.com/suyog44/InnerveX-Final.git
   - cd InnerveX-Final

2. Create a Python virtual environment
   - python3 -m venv .venv
   - source .venv/bin/activate

3. Install Python dependencies
   - pip install -r requirements.txt
   If you did not create requirements.txt, install packages manually:
   - pip install numpy opencv-python pillow pyyaml matplotlib ultralytics

4. System / Pi-specific:
   - Update system packages:
     - sudo apt update && sudo apt upgrade -y
   - Install Picamera2 and supporting packages (Raspberry Pi):
     - Follow the official Picamera2 installation guide: https://www.raspberrypi.com/documentation/computers/camera_software.html (installation may require apt packages and libcamera)
   - Enable I2C:
     - sudo raspi-config ? Interface Options ? I2C ? Enable
   - Install Adafruit Blinka (if not installed) and MLX90640 driver:
     - pip install adafruit-circuitpython-mlx90640
     - pip install adafruit-blinka

5. Place YOLO weights (see next section)

YOLO weights
- The GUIs expect a YOLOv8 weights file (by default `yolov8n.pt` or `yolov8n.pt` inside models/). Obtain weights by:
  - Download pretrained yolov8n: https://github.com/ultralytics/ultralytics (or use the Ultralytics model hub from code)
  - Place the file at one of:
    - repo root: ./yolov8n.pt
    - ./models/yolov8n.pt
  - Alternatively adapt the code to point to your weights path (search for `YOLO_WEIGHTS` in apps/).

Configuration files

1) Stereo homography YAML (data/stereo_alignment_matrix.yaml)
- Used to warp/align secondary (IMX500) frames into the master (IMX519) frame.
- Expected fields:
  - homography_secondary_to_master: 3x3 matrix (list of 9 numbers or nested lists)
  - target_shape: [width, height]
- Example (JSON-style YAML):
  homography_secondary_to_master:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  target_shape: [640, 640]

If the YAML is missing, the apps fall back to identity homography and default sizes.

2) MLX manual alignment JSON (data/mlx_manual_align.json)
- Stores a small alignment transform for MLX thermal overlays (dx, dy, scale)
- Example:
  {
    "dx": 0,
    "dy": 0,
    "scale": 1.0
  }
- If missing, the apps fall back to sensible defaults.

Running the GUIs
- All GUIs are Tkinter apps. They will attempt to open Picamera2 devices and MLX90640 over I2C.

Common commands:
- python3 apps/spectra_main.py
  - Full dashboard with YOLO, thermal fusion, interactive controls and a dashboard window.
- python3 apps/spectra_gui_2.py
  - Lightweight 3-view demo (IMX500 -> IMX519 mapped, IMX519 master, MLX thermal)
- python3 apps/multispectral_fusion_app_gui.py
  - Fusion-focused GUI (similar configuration)
- python3 detect4c_gui.py
  - Another dashboard application supporting fusion and tiled displays

Notes while running
- The apps search for cameras by name: "imx519" (MASTER_NAME) and "imx500" (SECONDARY_NAME). If camera names are not found they fall back to indices 0 and 1.
- The MLX90640 is initialized with a default refresh rate (e.g., 16Hz). Make sure the sensor is powered and I2C enabled.
- If YOLO weights are missing, the software may still run but detections will not work; look for console messages about weight paths.

Detection logging & reporting
- Detections are appended to a CSV (data/detections.csv). The apps try to ensure this file exists.
- parser.py aggregates data/detections.csv into visualization PNGs and builds data/report.html containing a SPECTRA report.
  - Example usage:
    - python3 parser.py --csv data/detections.csv --out data/report.html

Test utilities
- test/opencv_bothcam.py — simple OpenCV viewer showing two Picamera2 streams side-by-side (IMX500 and IMX519). Useful to verify camera indices, preview sizes and frame capture.

Troubleshooting
- Camera not detected:
  - Run a camera test (e.g., raspistill or the test script)
  - Confirm Picamera2 installed and libcamera is working
  - The apps print detected cameras via `Picamera2.global_camera_info()`; inspect that console output
- MLX90640 read errors:
  - Ensure I2C is enabled and wiring is correct
  - Try running a minimal MLX90640 read script to confirm the sensor
- YOLO errors:
  - Ensure ultralytics is installed and version matches your environment
  - Confirm yolov8n.pt exists and is readable
- GUI freezes or high CPU:
  - YOLO inference on CPU may be slow. Use smaller weights (yolov8n) or run on hardware with acceleration.
  - Reduce preview resolution in config constants (TW/TH) to lighten load.

Development notes / code structure (quick)
- apps/spectra_main.py: main dashboard, uses modules.vision_yolo and modules.draw_utils (if present)
- apps/spectra_gui_2.py: demo app with Detector class using ultralytics.YOLO
- detect4c_gui.py: tiled multispectral dashboard with fusion, thermal normalization, and MLX alignment helpers
- heatmap.py: thermal heatmap utilities
- parser.py: report builder
- data/: runtime data and generated artifacts

If you modify or extend the code:
- Keep alignment files in data/
- Keep YOLO weights path consistent with code expectations or change the `YOLO_WEIGHTS` constant
- Consider splitting heavy processing (YOLO inference) into a separate thread/process if you need UI responsiveness improvements

Contributing
- Contributions are welcome. Typical workflow:
  - Fork -> branch -> PR
  - Keep changes focused (e.g., hardware support, drift correction, model integration)
  - Add or update documentation for new features or hardware instructions

License
- MIT License. See LICENSE file in the repository.

Acknowledgements
- Builds on Ultralytics YOLO for detection
- Uses Adafruit libraries for MLX90640
- Picamera2 for camera capture on Raspberry Pi

---