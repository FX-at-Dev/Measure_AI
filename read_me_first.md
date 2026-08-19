# Read Me First

## Python version

This project was checked with **Python 3.14.6**.

## Required libraries

These are the direct third-party libraries imported by the project scripts and their versions in the project environment:

| Library | Version | Used for |
| --- | --- | --- |
| `opencv-python` | `5.0.0.93` | Webcam access, image processing, and display windows |
| `numpy` | `2.4.6` | Image and numeric array operations |
| `mediapipe` | `1.0.1` | Pose-landmark detection for limb and height estimates |
| `ultralytics` | `8.4.121` | YOLO person segmentation in `auto_measure.py` |

`os`, `sys`, `math`, `time`, and `urllib.request` are Python standard-library modules and do not need installation.

## Install

From the project folder, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the exact versions used in this environment:

```powershell
python -m pip install opencv-python==5.0.0.93 numpy==2.4.6 mediapipe==1.0.1 ultralytics==8.4.121
```

To run the main application:

```powershell
python Measure_AI.py
```

The `auto_measure.py` script additionally requires the included `yolov8n-seg.pt` model. The main application uses the included `pose_landmarker.task` model; if it is missing, it attempts to download it when first run.
