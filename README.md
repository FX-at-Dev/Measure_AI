# Measure AI

Measure AI is a desktop application for estimating body dimensions from photos and suggesting cricket-gear sizes. It collects a small set of body photos — captured live or imported from your existing gallery — calibrates image scale from an A4 sheet, detects MediaPipe pose, hand, and face landmarks, estimates limb lengths, height, hand length and head circumference, and produces sizing guidance for cricket equipment.

> **Prototype notice:** Measurements and recommendations are estimates, not a substitute for in-person fitting or professional measurement. The current limb correction factors were derived from limited testing and should be validated before relying on them for purchasing decisions. Head circumference in particular is *modelled* from face width rather than measured — always tape-measure before buying a helmet.

## Features

- One window for the whole pipeline: collect photos → calibrate → measure → recommend
- Each photo can come from the **webcam** (live preview with countdown capture) or the **gallery**
- Image-quality checks that reject overly blurry, dark, or overexposed captures
- A4-paper calibration to estimate pixels per centimetre, with the detector's candidate boxes drawn on the photo so you can see what it considered
- MediaPipe pose detection for upper arms, forearms, thighs, shins, and estimated height
- MediaPipe hand detection for hand length and breadth
- MediaPipe face detection plus an ellipse model for estimated head circumference
- Automatic pose-guided cropping so a full-body photo still yields hand and head measurements
- Manual click-to-mark override for the A4 sheet and every body measurement, with a cursor magnifier for precision
- Cricket sizing for batting pads, arm guards, thigh guards, bats, batting gloves, and helmets
- Detection runs on a worker thread, so the window stays responsive while photos are measured
- Standalone automatic and manual height-measurement experiments using an A4 reference
- Landmark visualisation utility for inspecting pose-detection confidence

## Requirements

- Python 3.9 or later
- A webcam for live capture — not needed if you import every photo from the gallery
- An A4 sheet of paper, held flat and facing the camera, for calibration
- A desktop environment (the application is a native window)

The application uses PySide6, OpenCV, NumPy, and MediaPipe. The automated height experiment also uses Ultralytics YOLO.

## Getting started

Clone the repository and create an isolated Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install PySide6 opencv-python numpy mediapipe ultralytics
```

Run the application from the repository root:

```powershell
python Measure_AI.py
```

## Using the application

The window has a list of steps down the left and the active step on the right. Each step carries a status mark:

| Mark | Meaning |
| --- | --- |
| `✓` | Photo present and measured |
| `!` | Photo present but nothing could be measured from it |
| `–` | Skipped |
| `·` | No photo yet |

The path of the file each step is reading is shown under the canvas.

### Where photos live

Gallery photos are used from wherever you picked them; they are never copied or altered. Only the camera writes files, into `captured_photos/<step>.jpg`.

Your choices are recorded in `captured_photos/session.json`, so closing the application and reopening it restores the same set of photos without re-shooting. If a file has since been moved or deleted, that step simply comes back empty.

### Supplying a photo

Each step offers four buttons:

| Button | What it does |
| --- | --- |
| **Use camera** | Opens a live preview in the canvas. **Capture** starts a 3-second countdown, then runs the quality check — a rejected frame tells you why and lets you retry in place. |
| **Choose from gallery** | A file picker. The photo is read **where it is** — nothing is copied or modified, so choosing a photo can never overwrite another. Quality problems are only a *warning* here, with the option to use it anyway. |
| **Mark by hand** | Click the points yourself. See [When the detection is wrong](#when-the-detection-is-wrong). |
| **Skip** | The gear depending on that photo simply won't be sized. |

Setting a photo measures it immediately in the background; the result appears under the canvas. **Re-detect** runs it again.

Photos are collected in this order, and each one unlocks specific recommendations:

| Photo | What to show | Unlocks |
| --- | --- | --- |
| `calibration.jpg` | An A4 sheet held flat, facing the camera | The pixels-per-cm scale everything else depends on |
| `arms.jpg` | Arms out to the side, elbows and wrists visible | Arm guards |
| `legs.jpg` | Legs slightly apart, hips/knees/ankles visible | Batting pads |
| `full_body_height.jpg` | Whole body, head to feet | Thigh guard, bat size |
| `hands.jpg` | One open palm flat to the camera | Batting gloves |
| `head.jpg` | Face straight on, whole head in frame | Helmet |

Do the calibration step first. Every other measurement is a pixel distance divided by the scale it produces, so nothing downstream can produce centimetres without it — the other steps will say so if you try. When the scale changes, everything measured automatically is re-measured against it.

For best results, keep the subject and A4 sheet in the same plane, use even lighting, and ensure shoulders, elbows, wrists, hips, knees, ankles, and feet are visible.

### The Results page

The last item in the sidebar collects everything: a table of measurements with the source of each (`automatic` or `marked by hand`), and the cricket sizes derived from them. **Re-measure everything** re-runs every automatic measurement, leaving anything you marked by hand alone.

### When the detection is wrong

Automatic detection fails in two ways: loudly (no person found) and quietly (a confident elbow placed 30 px off, or the paper detector locking onto a bright wall tile). Both are recoverable — **Mark by hand** on any step lets you click the points yourself.

On the calibration step, the detector's candidates are drawn on the photo — green for the one it accepted, red for those it rejected — so a wrong lock is visible rather than silent.

While marking:

| Action | Effect |
| --- | --- |
| Left click | Place the next point |
| **Undo** / `u` | Remove the last point |
| **Reset** / `r` | Start over |
| **Accept** | Enabled once every point is placed |
| **Cancel** | Discard, keeping the automatic value |

A banner across the top of the canvas names the point you are placing. A magnifier follows the cursor showing full-resolution pixels with a crosshair — this matters, because a 4032 px photo displayed in a 700 px canvas would otherwise round every click to the nearest six pixels of real image. Clicks are converted back to original-image coordinates, so they stay consistent with the calibration scale.

You can click the A4 corners in any order and any direction — they are sorted by angle around their centre. If opposite edges disagree by more than 10%, the sheet was tilted rather than flat-on, and you are warned because every measurement inherits that error.

What you can mark by hand:

| Measurement | Points to click |
| --- | --- |
| Calibration | The four corners of the A4 sheet |
| Arms | Shoulder → elbow → wrist (you pick the side) |
| Legs | Hip → knee → ankle (you pick the side) |
| Height | Top of the head → the ground at the heels |
| Hand | Wrist crease → middle fingertip, then the index and little finger knuckles |
| Head | The widest point on each side of the head |

Marking arms or legs asks which side you are marking; the rest have a fixed set of points. Manually marked values are labelled `marked by hand` on the Results page, and **Re-measure everything** leaves them alone.

Two deliberate differences from the automatic path:

- **No correction factor is applied to manual measurements.** `CORRECTION_FACTORS` was fitted to MediaPipe's automatic joint landmarks; a human clicking a knee has a different error profile, and scaling a hand-placed point by 1.34× would simply inflate it. If a manual and an automatic result disagree, that gap *is* the correction factor being tested — worth recording.
- **Manual height is strictly better than automatic**, not merely a fallback. The pose landmarker has no crown landmark, so it adds `HEAD_TOP_OFFSET_CM`, a 12 cm population average. Clicking the actual top of the head removes that assumption entirely.

### Full-body photos for hands and head

The hand and face detectors need their subject to occupy a reasonable share of the frame; on a full-body shot the face may be only a couple of hundred pixels wide and detection fails outright. When that happens the app falls back to locating the head or wrist with the pose landmarker and re-running the detector on a crop of that region. The crop is not resized, so the calibration scale still applies — but the result is flagged as less accurate, and a genuine close-up is always better.

## Other scripts

| Script | Purpose | Run |
| --- | --- | --- |
| `Measure_AI.py` | The application — capture, calibration, measurement, manual marking, and recommendations | `python Measure_AI.py` |
| `auto_measure.py` | YOLO-based automatic height experiment using `IMG_1216.jpg` and an A4 sheet | `python auto_measure.py` |
| `Manual_measure_ai.py` | The original click-to-measure experiment for height; this approach is now built into `Measure_AI.py` for every measurement | `python Manual_measure_ai.py` |
| `measure_limbs.py` | Standalone MediaPipe arm and leg measurement experiment | `python measure_limbs.py` |
| `visualize_landmarks.py` | Writes landmark overlays to debug images | `python visualize_landmarks.py` |
| `phase1_step4_quality_check.py` | Standalone webcam capture and image-quality-check experiment | `python phase1_step4_quality_check.py` |

The repository includes the MediaPipe and YOLO model files used by these scripts. `pose_landmarker.task`, `face_landmarker.task`, and `hand_landmarker.task` are each downloaded automatically on first use if absent.

## Example output

The Results page shows measurements alongside where each came from:

| Measurement | Value | Source |
| --- | --- | --- |
| `left_forearm_cm` | 25.4 cm | automatic |
| `left_shin_cm` | 42.1 cm | automatic |
| `height_cm` | 176.0 cm | marked by hand |
| `hand_length_cm` | 19.2 cm | automatic |
| `head_width_cm` | 14.8 cm | automatic |
| `head_circumference_cm` | ~55.4 cm | estimated from width (cephalic index 80) |

and the sizes derived from them:

| Item | Recommended size |
| --- | --- |
| Batting Pads | Adult (41-44cm knee-to-instep) |
| Arm Guards | Adult (rough estimate — no published chart found, verify in person) |
| Thigh Guard | Medium (41cm pad) — Remfry chart, height 170-185cm |
| Bat Size | Academy / Small Short Handle — 175-188cm (Chase Cricket chart) |
| Batting Gloves | Adult / Mens (rough banding — verify in person before buying) |
| Helmet | Medium (approx 55-58cm — verify in person before buying) |

Skipped photos produce a "could not measure" row rather than a guess.

## How the estimates are derived

Not every number carries the same confidence, and the source comments in `Measure_AI.py` say so explicitly:

| Measurement | Basis | Confidence |
| --- | --- | --- |
| Limb segments | Pixel distance ÷ scale, times a per-segment correction factor | Correction factors came from **one** tape-measured subject — a documented stopgap |
| Height | Eye level to lowest foot, plus a 12 cm eye-to-crown population average | No correction applied; a fixed factor was tried and **removed** after it made a second subject's result worse |
| Hand length | Wrist to middle fingertip | Raw, uncorrected — no ground-truth data exists to justify a factor |
| Head circumference | Face width → skull breadth → skull length via cephalic index 80 → ellipse perimeter | **Modelled, not measured.** An individual with a cephalic index of 75 or 85 will be off by roughly 2 cm — a whole helmet size |

Sizing bands cite a published retailer chart where one was found (batting pads, thigh guards, bat size). Arm guards, gloves, and helmets use generic banding and say so in their output, rather than presenting an invented chart as authoritative.

Marking a measurement by hand replaces the row's basis with your own clicks and drops the correction factor — see [When the detection is wrong](#when-the-detection-is-wrong).

Above all of these sits the calibration scale. It is the one number every measurement is multiplied by, and its accuracy depends on the A4 sheet being flat-on and the *same distance from the camera as the body part being measured*. Moving the sheet from chest height to the edge of the frame changed one test subject's raw height by about 6 cm.

## Project layout

```text
Measure_AI.py                 Main application
auto_measure.py               Automatic-height prototype
Manual_measure_ai.py          Manual-height prototype
measure_limbs.py              Limb-measurement utilities
visualize_landmarks.py        Landmark inspection utility
pose_landmarker.task          MediaPipe pose model
face_landmarker.task          MediaPipe face model (head width)
hand_landmarker.task          MediaPipe hand model (downloaded on first use)
captured_photos/              Camera captures, plus session.json
Test/                         Sample images
```

`Measure_AI.py` is organised into numbered sections. Sections 1–6 are the engine: they take paths and numbers, return numbers, and know nothing about the interface. Section 7 is the PySide6 GUI and is the only part that touches widgets. Detection runs on a worker thread so the window stays responsive.

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
