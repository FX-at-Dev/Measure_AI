# Measure AI

Measure AI is a webcam and photo-based prototype for estimating body dimensions and suggesting cricket-gear sizes. It captures a small set of body photos, calibrates image scale from an A4 sheet, detects MediaPipe pose landmarks, estimates limb lengths and height, and produces sizing guidance for selected cricket equipment.

> **Prototype notice:** Measurements and recommendations are estimates, not a substitute for in-person fitting or professional measurement. The current limb correction factors were derived from limited testing and should be validated before relying on them for purchasing decisions.

## Features

- Guided webcam capture for hands, arms, head, legs, and full-body photos
- Image-quality checks that reject overly blurry, dark, or overexposed captures
- A4-paper calibration to estimate pixels per centimetre
- MediaPipe pose detection for upper arms, forearms, thighs, shins, and estimated height
- Cricket sizing recommendations for batting pads, arm guards, thigh guards, and bats
- Standalone automatic and manual height-measurement experiments using an A4 reference
- Landmark visualisation utility for inspecting pose-detection confidence

## Requirements

- Python 3.9 or later
- A webcam for the interactive capture and calibration flows
- An A4 sheet of paper, held flat and facing the camera, for calibration
- A supported desktop environment that can display OpenCV windows

The main application uses OpenCV, NumPy, and MediaPipe. The automated height experiment also uses Ultralytics YOLO.

## Getting started

Clone the repository and create an isolated Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python numpy mediapipe ultralytics
```

Run the main application from the repository root:

```powershell
python Measure_AI.py
```

Choose one of the menu options:

1. **Capture body part photos** — press `Space` to start the countdown for each requested photo; press `Q` to quit.
2. **Calibrate** — use the saved `captured_photos/calibration.jpg` image if it exists, otherwise hold an A4 sheet in front of the webcam and press `Space` to capture it. Press `Y` to accept a successful calibration.
3. **Measure and recommend** — recalibrates, measures the saved arms, legs, and full-body images, then prints the available gear recommendations.

For best results, keep the subject and A4 sheet in the same plane, use even lighting, and ensure shoulders, elbows, wrists, hips, knees, ankles, and feet are visible. The program needs these captured files before measurement:

```text
captured_photos/
├── arms.jpg
├── legs.jpg
├── full_body_height.jpg
└── calibration.jpg
```

The capture flow creates the first three files. Save a suitable A4 calibration photo as `captured_photos/calibration.jpg`, or use the live calibration flow when that file is absent.

## Other scripts

| Script | Purpose | Run |
| --- | --- | --- |
| `Measure_AI.py` | Primary interactive capture, calibration, measurement, and recommendation workflow | `python Measure_AI.py` |
| `auto_measure.py` | YOLO-based automatic height experiment using `IMG_1216.jpg` and an A4 sheet | `python auto_measure.py` |
| `Manual_measure_ai.py` | Click four A4 corners, then the head and heels to calculate height manually | `python Manual_measure_ai.py` |
| `measure_limbs.py` | Standalone MediaPipe arm and leg measurement experiment | `python measure_limbs.py` |
| `visualize_landmarks.py` | Writes landmark overlays to debug images | `python visualize_landmarks.py` |
| `phase1_step4_quality_check.py` | Standalone webcam capture and image-quality-check experiment | `python phase1_step4_quality_check.py` |

The repository includes the MediaPipe and YOLO model files used by these scripts. If `pose_landmarker.task` is missing, the main and limb-measurement workflows attempt to download it on first use.

## Example output

After a successful measurement, the main program prints measurements in centimetres and the recommendations that can be derived from them:

```text
--- Measurements ---
left_forearm_cm: 25.4 cm
left_shin_cm: 42.1 cm
height_cm: 176.0 cm

--- Cricket Gear Recommendations ---
batting_pads: Adult (41-44cm knee-to-instep)
arm_guards: Adult (rough estimate -- verify in person)
thigh_guard: Medium (41cm pad)
bat_size: Academy / Small Short Handle
```

Some recommendations intentionally remain unavailable because the current pipeline does not yet measure hand length or head circumference.

## Project layout

```text
Measure_AI.py                 Main application
auto_measure.py               Automatic-height prototype
Manual_measure_ai.py          Manual-height prototype
measure_limbs.py              Limb-measurement utilities
visualize_landmarks.py        Landmark inspection utility
captured_photos/              Captured input images
Test/                         Sample images
```

## Help

Start with the inline prompts from `Measure_AI.py` and inspect `visualize_landmarks.py` when a pose is not detected or has low-confidence landmarks. The debug overlays it produces help identify framing problems.

For questions, bugs, or enhancement ideas, open an issue in the repository that hosts this project. Include the script you ran, your operating system and Python version, the command used, and—where appropriate—a non-sensitive sample image or a description of the lighting and camera setup.

## Contributing

Contributions are welcome, especially improvements to calibration accuracy, validation data, pose reliability, and cricket-equipment sizing sources.

Before submitting a change:

1. Keep the main workflow runnable with `python Measure_AI.py`.
2. Test the specific capture or measurement flow affected by your change.
3. Avoid committing private captured photos or generated debug images unless they are intentional, anonymised test fixtures.
4. Describe the dataset, measurement protocol, or sizing source behind any accuracy-related change.

There is no separate contribution guide yet; use a focused pull request with a clear description of the behaviour changed and how it was tested.

## Maintainers

Maintainer details have not been published in this workspace. Repository owners maintain the project and review contributions through the hosting repository.

## License

No license file is currently included. Contact the repository owner before reusing or redistributing the code.
