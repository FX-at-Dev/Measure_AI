"""
Measure_AI.py -- Cricket gear sizing assistant, v1.0

Pipeline:
  1. CAPTURE  -- open camera, prompt through body parts, countdown capture,
                 quality check (blur/brightness) before accepting a photo.
  2. CALIBRATE -- detect an A4 sheet held in a photo to compute pixels_per_cm.
  3. MEASURE  -- MediaPipe Pose landmarks -> segmented arm/leg lengths,
                 with empirical correction factors applied.
  4. RECOMMEND -- map measurements to cricket gear sizes, using real
                 published sizing standards where we have them.

Run with no arguments for the interactive menu.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
)
import time
import sys
import os
import math
import urllib.request

# =============================================================================
# SECTION 1: CAPTURE (from phase1_step4_quality_check.py)
# =============================================================================

BODY_PARTS = ["hands", "arms", "head", "legs", "full_body_height"]
OUTPUT_DIR = "captured_photos"

COUNTDOWN_SECONDS = 3
BLUR_THRESHOLD = 100
DARK_THRESHOLD = 50
BRIGHT_THRESHOLD = 220


def check_photo_quality(image):
    """Returns (is_good: bool, reason: str or None)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < BLUR_THRESHOLD:
        return False, f"Too blurry (sharpness={laplacian_var:.0f}, need >{BLUR_THRESHOLD})"

    brightness = gray.mean()
    if brightness < DARK_THRESHOLD:
        return False, f"Too dark (brightness={brightness:.0f}, need >{DARK_THRESHOLD})"
    if brightness > BRIGHT_THRESHOLD:
        return False, f"Too bright/overexposed (brightness={brightness:.0f}, need <{BRIGHT_THRESHOLD})"

    return True, None


def run_body_part_capture():
    """Cycles through BODY_PARTS, countdown capture + quality check for each."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    current_index = 0
    countdown_start_time = None
    warning_message = None
    warning_shown_time = None
    WARNING_DISPLAY_SECONDS = 2

    print("Instructions: SPACE = start 3-second countdown, Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        part_name = BODY_PARTS[current_index] if current_index < len(BODY_PARTS) else None
        display_frame = frame.copy()

        if part_name is None:
            cv2.putText(display_frame, "All body parts captured! Press Q to exit.",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif countdown_start_time is None:
            cv2.putText(display_frame, f"Show your: {part_name.upper()}  (SPACE to start countdown)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            elapsed = time.time() - countdown_start_time
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining > 0:
                cv2.putText(display_frame, f"Get ready: {part_name.upper()}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, str(int(remaining) + 1),
                            (280, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)
            else:
                is_good, reason = check_photo_quality(frame)
                if is_good:
                    filename = os.path.join(OUTPUT_DIR, f"{part_name}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Captured: {filename}")
                    current_index += 1
                else:
                    print(f"Rejected capture for '{part_name}': {reason}")
                    warning_message = f"Retry: {reason}"
                    warning_shown_time = time.time()
                countdown_start_time = None

        if warning_message is not None:
            if time.time() - warning_shown_time < WARNING_DISPLAY_SECONDS:
                cv2.putText(display_frame, warning_message,
                            (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                warning_message = None

        cv2.imshow("Body Measurement Capture", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and countdown_start_time is None and part_name is not None:
            countdown_start_time = time.time()
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {current_index} of {len(BODY_PARTS)} body parts captured.")


# =============================================================================
# SECTION 2: CALIBRATION (from Measure_AI.py's paper detection)
# =============================================================================

PAPER_REAL_WIDTH_CM = 21.0
PAPER_REAL_HEIGHT_CM = 29.7
MIN_AREA_PX = 8000  # absolute floor -- see note in earlier debugging: percentage-based was wrong
EXPECTED_ASPECT_RATIO = PAPER_REAL_WIDTH_CM / PAPER_REAL_HEIGHT_CM
ASPECT_RATIO_TOLERANCE = 0.15

CALIBRATION_IMAGE_PATH = "captured_photos/calibration.jpg"


def _evaluate_paper_candidate(contour, image_width, image_height, min_area_px):
    area = cv2.contourArea(contour)
    box = cv2.boundingRect(contour)

    if area < min_area_px:
        return None, "too small", box, area

    hull = cv2.convexHull(contour)
    box = cv2.boundingRect(hull)

    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.03 * perimeter, True)
    if len(approx) < 4 or len(approx) > 6:
        return None, f"{len(approx)} corners, not roughly 4", box, area

    x, y, w, h = box
    border_margin = 5
    if x <= border_margin or y <= border_margin or \
       (x + w) >= (image_width - border_margin) or \
       (y + h) >= (image_height - border_margin):
        return None, "touches the frame edge", box, area

    rect = cv2.minAreaRect(hull)
    (rect_width, rect_height) = rect[1]
    if rect_width == 0 or rect_height == 0:
        return None, "invalid dimensions", box, area

    short_side = min(rect_width, rect_height)
    long_side = max(rect_width, rect_height)
    detected_ratio = short_side / long_side

    if abs(detected_ratio - EXPECTED_ASPECT_RATIO) > ASPECT_RATIO_TOLERANCE:
        return None, f"wrong proportions (ratio {detected_ratio:.2f}, expected ~{EXPECTED_ASPECT_RATIO:.2f})", box, area

    pixels_per_cm = short_side / PAPER_REAL_WIDTH_CM
    return pixels_per_cm, None, box, area


def detect_paper_and_get_scale(image, return_all_candidates=False):
    """Returns (pixels_per_cm, reason, debug_box[, candidates])."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (None, "No bright region found at all", None, []) if return_all_candidates \
            else (None, "No bright region found at all", None)

    image_height, image_width = image.shape[:2]
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    SEARCH_CANDIDATES = 30
    DISPLAY_CANDIDATES = 5

    all_candidates = []
    winner = None

    for i, contour in enumerate(contours_sorted[:SEARCH_CANDIDATES]):
        pixels_per_cm, reason, box, area = _evaluate_paper_candidate(contour, image_width, image_height, MIN_AREA_PX)
        if i < DISPLAY_CANDIDATES:
            all_candidates.append({"box": box, "reason": reason, "pixels_per_cm": pixels_per_cm})
        if pixels_per_cm is not None and winner is None:
            winner = (pixels_per_cm, None, box)
            break

    if winner:
        result = winner
    else:
        first = all_candidates[0]
        result = (None, first["reason"], first["box"])

    if return_all_candidates:
        return result[0], result[1], result[2], all_candidates
    return result


def calibrate_from_saved_image(image_path):
    """Runs paper detection on an existing photo file."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image at {image_path}")
        return None

    pixels_per_cm, reason, box, candidates = detect_paper_and_get_scale(image, return_all_candidates=True)

    display_image = image.copy()
    for cand in candidates:
        if cand["box"] is None:
            continue
        x, y, w, h = cand["box"]
        passed = cand["pixels_per_cm"] is not None
        color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)

    if pixels_per_cm is not None:
        print(f"SUCCESS: {pixels_per_cm:.2f} px/cm")
        status_text, status_color = f"SUCCESS: {pixels_per_cm:.2f} px/cm", (0, 255, 0)
    else:
        print(f"FAILED: {reason}")
        status_text, status_color = f"FAILED: {reason}", (0, 0, 255)

    cv2.putText(display_image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    print("Showing result -- press any key in the image window to close.")
    cv2.imshow("Calibration From Saved Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pixels_per_cm


def run_calibration_capture():
    """Live capture-first-then-check calibration flow."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return None

    print("Hold your A4 sheet flat, facing the camera. SPACE = capture and check, Q = quit.")

    final_pixels_per_cm = None
    frozen_frame = None
    review_message = None
    review_color = (255, 255, 255)
    review_candidates = []

    while True:
        if frozen_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
            cv2.putText(display_frame, "SPACE = capture photo and check for paper",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Paper Calibration Capture", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                frozen_frame = frame.copy()
                pixels_per_cm, reason, box, candidates = detect_paper_and_get_scale(
                    frozen_frame, return_all_candidates=True)
                review_candidates = candidates
                if pixels_per_cm is not None:
                    final_pixels_per_cm = pixels_per_cm
                    review_message = f"SUCCESS: {pixels_per_cm:.2f} px/cm  (Y = accept, SPACE = retake)"
                    review_color = (0, 255, 0)
                else:
                    final_pixels_per_cm = None
                    review_message = f"FAILED: {reason}  (SPACE = retake)"
                    review_color = (0, 0, 255)
            if key == ord('q'):
                break
        else:
            display_frame = frozen_frame.copy()
            for cand in review_candidates:
                if cand["box"] is None:
                    continue
                x, y, w, h = cand["box"]
                passed = cand["pixels_per_cm"] is not None
                color = (0, 255, 0) if passed else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, review_message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, review_color, 2)
            cv2.imshow("Paper Calibration Capture", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                frozen_frame = None
                review_message = None
            if key == ord('y') and final_pixels_per_cm is not None:
                print(f"Calibration accepted: {final_pixels_per_cm:.2f} px/cm")
                break
            if key == ord('q'):
                final_pixels_per_cm = None
                break

    cap.release()
    cv2.destroyAllWindows()
    return final_pixels_per_cm


def get_calibration():
    """Auto-detect saved calibration photo, else open camera."""
    if os.path.exists(CALIBRATION_IMAGE_PATH):
        print(f"Found existing photo at {CALIBRATION_IMAGE_PATH} -- using it (no camera opened).")
        return calibrate_from_saved_image(CALIBRATION_IMAGE_PATH)
    else:
        print(f"No photo found at {CALIBRATION_IMAGE_PATH} -- opening live camera instead.")
        return run_calibration_capture()


# =============================================================================
# SECTION 3: POSE MEASUREMENT (from measure_limbs.py)
# =============================================================================

POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_eye": 2, "right_eye": 5,
    "left_foot_index": 31, "right_foot_index": 32,
}

MIN_VISIBILITY = 0.5

# --- TEMPORARY empirical correction factors ---
# Raw pixel-based measurements consistently come out too SHORT, and by a
# DIFFERENT amount per segment (worse near frame edges -- likely lens
# distortion). These multipliers were derived from ONE real test subject's
# tape-measured lengths vs our computed lengths. This is a stopgap, not a
# validated general fix -- replace with a trained correction model (or
# real lens calibration) once more (photo, true measurement) pairs exist.
CORRECTION_FACTORS = {
    "upper_arm": 1.34,
    "forearm": 1.17,
    "thigh": 1.20,
    "shin": 1.37,
    # Height correction: DISABLED (set to 1.0 = no correction).
    # We tried deriving a fixed factor from one test (176.4cm real vs
    # 161.4cm raw -> 1.093x), but testing against two more real cases
    # showed the raw error isn't a consistent, predictable bias:
    #   - Same person, paper calibrated at frame edge instead of center
    #     chest: raw changed from 161.4 to 155.5cm for the SAME person.
    #     Calibration paper POSITION measurably changes the raw error,
    #     which a single fixed multiplier can't account for.
    #   - A second person: raw was 173.8cm vs their real 170cm (already
    #     close, ~2% over). Applying the 1.093x "fix" pushed it to a
    #     190cm final answer -- WORSE than doing nothing.
    # A constant correction factor assumes a stable, repeatable bias.
    # The evidence here says that assumption is false for height --
    # applying an unproven correction risks making some users' results
    # worse, not better. Left at 1.0 until enough consistent-protocol
    # ground-truth data exists to justify re-enabling this honestly.
    "height": 1.0,
}


def _ensure_pose_model_downloaded():
    if not os.path.exists(POSE_MODEL_PATH):
        print("Downloading pose landmarker model (one-time, ~30MB)...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("Model downloaded.")


def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def _get_pixel_coords(landmark, image_width, image_height):
    return (landmark.x * image_width, landmark.y * image_height)


def detect_pose_landmarks(image_path):
    _ensure_pose_model_downloaded()

    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        print(f"Error: could not load image at {image_path}")
        return None, None, None

    image_height, image_width = bgr_image.shape[:2]
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
        if not result.pose_landmarks:
            print(f"No person detected in {image_path}")
            return None, None, None

        raw_landmarks = result.pose_landmarks[0]
        landmarks_dict = {}
        for name, index in LANDMARKS.items():
            lm = raw_landmarks[index]
            landmarks_dict[name] = {
                "pixel_coords": _get_pixel_coords(lm, image_width, image_height),
                "visibility": lm.visibility,
            }
        return landmarks_dict, image_width, image_height


def measure_segment(landmarks_dict, point_a_name, point_b_name, pixels_per_cm, correction_key=None):
    point_a = landmarks_dict[point_a_name]
    point_b = landmarks_dict[point_b_name]
    if point_a["visibility"] < MIN_VISIBILITY or point_b["visibility"] < MIN_VISIBILITY:
        return None
    pixel_distance = euclidean_distance(point_a["pixel_coords"], point_b["pixel_coords"])
    raw_cm = pixel_distance / pixels_per_cm
    if correction_key is not None:
        return raw_cm * CORRECTION_FACTORS[correction_key]
    return raw_cm


def measure_arms_and_legs(arms_photo_path, legs_photo_path, pixels_per_cm):
    results = {}

    landmarks, w, h = detect_pose_landmarks(arms_photo_path)
    if landmarks:
        for side in ["left", "right"]:
            results[f"{side}_upper_arm_cm"] = measure_segment(
                landmarks, f"{side}_shoulder", f"{side}_elbow", pixels_per_cm, "upper_arm")
            results[f"{side}_forearm_cm"] = measure_segment(
                landmarks, f"{side}_elbow", f"{side}_wrist", pixels_per_cm, "forearm")

    landmarks, w, h = detect_pose_landmarks(legs_photo_path)
    if landmarks:
        for side in ["left", "right"]:
            results[f"{side}_thigh_cm"] = measure_segment(
                landmarks, f"{side}_hip", f"{side}_knee", pixels_per_cm, "thigh")
            results[f"{side}_shin_cm"] = measure_segment(
                landmarks, f"{side}_knee", f"{side}_ankle", pixels_per_cm, "shin")

    return results


# --- Height estimation ---
# MediaPipe Pose has no "top of skull" landmark -- its highest facial
# points are the eyes. We estimate the eye-to-crown gap using a general
# adult anthropometric average (~12cm from eye level to the top of the
# head). This is a population average, not measured from you specifically
# -- like the limb correction factors, it's a documented approximation to
# be refined with real ground-truth data over time, not a precise fit.
HEAD_TOP_OFFSET_CM = 12.0


def measure_height(landmarks_dict, pixels_per_cm):
    """
    Estimates full-body height from eye-level to foot, plus the
    estimated eye-to-crown offset. Returns height in cm, or None if
    the needed landmarks aren't reliably visible.
    """
    # Top reference: average eye position (both eyes preferred for
    # stability; each individually gated by visibility).
    eye_points = []
    for side in ["left", "right"]:
        eye = landmarks_dict.get(f"{side}_eye")
        if eye and eye["visibility"] >= MIN_VISIBILITY:
            eye_points.append(eye["pixel_coords"])
    if not eye_points:
        return None
    eye_y = sum(p[1] for p in eye_points) / len(eye_points)

    # Bottom reference: prefer foot_index (closer to the ground than the
    # ankle joint); fall back to ankle if foot_index isn't visible.
    bottom_points = []
    for side in ["left", "right"]:
        foot = landmarks_dict.get(f"{side}_foot_index")
        ankle = landmarks_dict.get(f"{side}_ankle")
        if foot and foot["visibility"] >= MIN_VISIBILITY:
            bottom_points.append(foot["pixel_coords"])
        elif ankle and ankle["visibility"] >= MIN_VISIBILITY:
            bottom_points.append(ankle["pixel_coords"])
    if not bottom_points:
        return None
    bottom_y = max(p[1] for p in bottom_points)  # largest y = lowest in image

    head_top_offset_px = HEAD_TOP_OFFSET_CM * pixels_per_cm
    top_of_head_y = eye_y - head_top_offset_px

    height_px = bottom_y - top_of_head_y
    return height_px / pixels_per_cm


def measure_height_from_photo(height_photo_path, pixels_per_cm):
    landmarks, w, h = detect_pose_landmarks(height_photo_path)
    if not landmarks:
        return None
    raw_height = measure_height(landmarks, pixels_per_cm)
    if raw_height is None:
        return None
    return raw_height * CORRECTION_FACTORS["height"]


# =============================================================================
# SECTION 4: CRICKET GEAR RECOMMENDATION
# =============================================================================
# Sizing bands sourced from published cricket retailer size guides where
# noted. Real-world charts mostly key off measurements we DON'T capture
# yet (overall height, hand length, head circumference) -- those
# categories are marked NOT_YET_AVAILABLE rather than guessed.
#
# Sources (checked against real retailer size guides):
#  - Batting pads: knee-to-instep. Junior 30-35cm, Adult 41-44cm
#    (cricket-hockey.com). We approximate knee-to-instep with our
#    knee-to-ankle ("shin") measurement -- close but not identical.
#  - Thigh guards: sized by OVERALL HEIGHT (Remfry: <170cm->Small,
#    170-185cm->Medium, >185cm->Large). Now usable via measure_height().
#  - Bat size: sized by OVERALL HEIGHT (Chase Cricket chart). Now usable.
#  - Batting gloves: hand length (wrist-to-fingertip). We don't measure
#    hand length yet (only forearm), so NOT_YET_AVAILABLE.
#  - Helmets: head circumference. Not measured yet, NOT_YET_AVAILABLE.

NOT_YET_AVAILABLE = "Not available yet -- needs a measurement this pipeline doesn't capture yet"


def recommend_batting_pad_size(shin_cm):
    """
    Batting pads are sized by knee-to-instep length. We approximate this
    with knee-to-ankle ("shin"). Real breakpoints: Junior 30-35cm,
    Adult 41-44cm (cricket-hockey.com). The 35-41cm gap between those
    two published tiers isn't covered by our source, so that middle
    band is OUR interpolation, not a verified published size --
    flagged as such below.
    """
    if shin_cm < 30:
        return "Small Junior (approx -- below published Junior range, verify in person)"
    elif shin_cm <= 35:
        return "Junior (30-35cm knee-to-instep)"
    elif shin_cm < 41:
        return "Youth / Small Adult (interpolated estimate -- not from a published chart, try before buying)"
    elif shin_cm <= 44:
        return "Adult (41-44cm knee-to-instep)"
    else:
        return "Large Adult (approx -- above published Adult range, verify in person)"


def recommend_arm_guard_size(forearm_cm):
    """
    No published numeric arm guard chart was found during research --
    brands (e.g. Moonwalkr) have size charts but didn't expose numeric
    cm breakpoints in what we could access. This uses forearm length
    with generic small/medium/large banding as a rough placeholder --
    treat as a starting point, not a verified fit.
    """
    if forearm_cm < 20:
        return "Junior (rough estimate -- no published chart found, verify in person)"
    elif forearm_cm < 25:
        return "Small Adult (rough estimate -- no published chart found, verify in person)"
    else:
        return "Adult (rough estimate -- no published chart found, verify in person)"


def recommend_thigh_guard_size(height_cm):
    """
    Thigh guards are sized by overall height, not thigh length.
    Source: Remfry thigh guard chart --
      <170cm -> Small (36cm pad), 170-185cm -> Medium (41cm pad),
      >185cm -> Large (45cm pad).
    """
    if height_cm < 170:
        return "Small (36cm pad) -- Remfry chart, height <170cm"
    elif height_cm <= 185:
        return "Medium (41cm pad) -- Remfry chart, height 170-185cm"
    else:
        return "Large (45cm pad) -- Remfry chart, height >185cm"


def recommend_bat_size(height_cm):
    """
    Bat size is sized by overall height. Source: Chase Cricket chart.
    Heights below are the LOWER bound of each tier.
    """
    tiers = [
        (188, "Short Handle (SH) -- 188cm+"),
        (175, "Academy / Small Short Handle -- 175-188cm"),
        (168, "Harrow -- 168-175cm"),
        (162.5, "Size 6 -- 162.5-168cm"),
        (157.5, "Size 5 -- 157.5-162.5cm"),
        (150, "Size 4 -- 150-157.5cm"),
        (145, "Size 3 -- 145-150cm"),
        (137, "Size 2 -- 137-145cm"),
        (129.5, "Size 1 -- 129.5-137cm"),
    ]
    for min_height, label in tiers:
        if height_cm >= min_height:
            return f"{label} (Chase Cricket chart)"
    return "Below Size 1 range -- consult a junior specialist retailer"


def recommend_gear(measurements, height_cm=None):
    """
    Takes the measurements dict from measure_arms_and_legs() (and
    optionally a separately-computed height_cm) and returns gear
    recommendations. Uses LEFT side measurements as representative
    (left/right should be close -- see the symmetry check built earlier).
    """
    recommendations = {}

    shin = measurements.get("left_shin_cm") or measurements.get("right_shin_cm")
    forearm = measurements.get("left_forearm_cm") or measurements.get("right_forearm_cm")

    recommendations["batting_pads"] = recommend_batting_pad_size(shin) if shin else "Could not measure shin"
    recommendations["arm_guards"] = recommend_arm_guard_size(forearm) if forearm else "Could not measure forearm"

    if height_cm:
        recommendations["thigh_guard"] = recommend_thigh_guard_size(height_cm)
        recommendations["bat_size"] = recommend_bat_size(height_cm)
    else:
        recommendations["thigh_guard"] = f"{NOT_YET_AVAILABLE} (needs: overall height)"
        recommendations["bat_size"] = f"{NOT_YET_AVAILABLE} (needs: overall height)"

    # Still gated -- needs hand length / head circumference, not yet measured.
    recommendations["batting_gloves"] = f"{NOT_YET_AVAILABLE} (needs: hand length, wrist-to-fingertip)"
    recommendations["helmet"] = f"{NOT_YET_AVAILABLE} (needs: head circumference)"

    return recommendations


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=== Measure_AI: Cricket Gear Sizing ===\n")
    print("1. Capture body part photos")
    print("2. Calibrate (paper detection)")
    print("3. Measure arms/legs + get gear recommendations")
    choice = input("Choose an option (1/2/3): ").strip()

    if choice == "1":
        run_body_part_capture()

    elif choice == "2":
        if len(sys.argv) > 1:
            calibrate_from_saved_image(sys.argv[1])
        else:
            get_calibration()

    elif choice == "3":
        pixels_per_cm = get_calibration()
        if not pixels_per_cm:
            print("Calibration failed -- cannot measure without it.")
            sys.exit(1)

        arms_path = os.path.join(OUTPUT_DIR, "arms.jpg")
        legs_path = os.path.join(OUTPUT_DIR, "legs.jpg")
        height_path = os.path.join(OUTPUT_DIR, "full_body_height.jpg")

        measurements = measure_arms_and_legs(arms_path, legs_path, pixels_per_cm)
        height_cm = measure_height_from_photo(height_path, pixels_per_cm)

        print("\n--- Measurements ---")
        for key, value in measurements.items():
            print(f"{key}: {value:.1f} cm" if value is not None else f"{key}: could not measure")
        print(f"height_cm: {height_cm:.1f} cm" if height_cm else "height_cm: could not measure "
              f"(HEAD_TOP_OFFSET_CM={HEAD_TOP_OFFSET_CM} is a population average -- treat with caution)")

        print("\n--- Cricket Gear Recommendations ---")
        recommendations = recommend_gear(measurements, height_cm)
        for item, size in recommendations.items():
            print(f"{item}: {size}")

    else:
        print("Invalid choice.")