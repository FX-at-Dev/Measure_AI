"""
Measure_AI.py -- Cricket gear sizing assistant, v2.0

Pipeline:
  1. COLLECT  -- for each required photo, either capture from the camera
                 (countdown + blur/brightness quality check) or import an
                 existing file from the gallery.
  2. CALIBRATE -- detect an A4 sheet held in a photo to compute pixels_per_cm.
  3. MEASURE  -- MediaPipe Pose/Hand/Face landmarks -> segmented arm & leg
                 lengths, height, hand length and head circumference, with
                 empirical correction factors applied.
  4. RECOMMEND -- map measurements to cricket gear sizes, using real
                 published sizing standards where we have them.

Any step can be marked by hand instead, by clicking the points on the
photo, for when the automatic detection is wrong.

Run `python Measure_AI.py` to open the application window.

Layout: sections 1-6 are the engine -- they take paths and numbers and
return numbers, and know nothing about the interface. Section 7 is the
PySide6 GUI and is the only part that touches widgets.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
)
from PySide6 import QtCore, QtGui, QtWidgets
import time
import sys
import os
import json
import math
import urllib.request

# =============================================================================
# SECTION 1: PHOTO COLLECTION -- CAMERA OR GALLERY
# (camera path originally from phase1_step4_quality_check.py)
# =============================================================================

OUTPUT_DIR = "captured_photos"

COUNTDOWN_SECONDS = 3
BLUR_THRESHOLD = 100
DARK_THRESHOLD = 50
BRIGHT_THRESHOLD = 220

# Every photo the pipeline can use, in the order we ask for them.
# "calibration" comes first on purpose: without a px/cm scale none of the
# other photos can be turned into centimetres, so there is no point
# collecting them if the A4 sheet can't be detected.
CAPTURE_STEPS = [
    {
        "key": "calibration",
        "label": "A4 sheet (calibration)",
        "instruction": "Hold an A4 sheet flat, facing the camera, at the same distance as your body will be",
        "unlocks": "the pixels-per-cm scale used by every other measurement",
    },
    {
        "key": "arms",
        "label": "Arms",
        "instruction": "Stand facing the camera, both arms out to the side, elbows and wrists visible",
        "unlocks": "upper arm + forearm lengths (arm guards)",
    },
    {
        "key": "legs",
        "label": "Legs",
        "instruction": "Stand facing the camera, legs slightly apart, hips/knees/ankles visible",
        "unlocks": "thigh + shin lengths (batting pads)",
    },
    {
        "key": "full_body_height",
        "label": "Full body",
        "instruction": "Stand well back so your whole body from head to feet is in frame",
        "unlocks": "overall height (thigh guard + bat size)",
    },
    {
        "key": "hands",
        "label": "Hand",
        "instruction": "Hold one open palm flat towards the camera, fingers straight and together",
        "unlocks": "hand length (batting gloves)",
    },
    {
        "key": "head",
        "label": "Head",
        "instruction": "Face the camera straight on, head upright, whole head in frame",
        "unlocks": "head circumference (helmet)",
    },
]

GALLERY_FILE_TYPES = [
    ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("All files", "*.*"),
]


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


def photo_path_for(part_key):
    """Where a CAMERA capture for this step is written."""
    return os.path.join(OUTPUT_DIR, f"{part_key}.jpg")


# Photos chosen from the gallery are used where they are -- we record the
# path here rather than copying the file into OUTPUT_DIR.
#
# An earlier version did copy, to <key>.jpg, so that a later run could
# find the photos again. That silently destroyed data: choosing any photo
# for a step overwrote whatever that step's slot already held, including
# the user's own originals if they picked one out of captured_photos/.
# It also crashed outright when the chosen file WAS the slot, because the
# same-file guard compared path strings and Windows treats
# "calibration.JPG" and "calibration.jpg" as one file with two spellings.
SESSION_FILE = os.path.join(OUTPUT_DIR, "session.json")


def same_file(path_a, path_b):
    """
    True if both paths name one file on disk.

    os.path.samefile is the only reliable test here: string comparison
    misses case differences on Windows, short (8.3) names, and symlinks.
    """
    try:
        return os.path.samefile(path_a, path_b)
    except OSError:  # one of them doesn't exist
        return False


def load_session_photos():
    """Restores {step_key: path} from the last run, dropping missing files."""
    photos = {}
    try:
        with open(SESSION_FILE, encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, ValueError):
        saved = {}

    valid_keys = {step["key"] for step in CAPTURE_STEPS}
    for key, path in saved.get("photos", {}).items():
        if key in valid_keys and isinstance(path, str) and os.path.exists(path):
            photos[key] = path

    # Anything the camera wrote in a previous run is still a valid source.
    for step in CAPTURE_STEPS:
        if step["key"] in photos:
            continue
        captured = photo_path_for(step["key"])
        if os.path.exists(captured):
            photos[step["key"]] = captured
    return photos


def save_session_photos(photos):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(SESSION_FILE, "w", encoding="utf-8") as handle:
            json.dump({"photos": photos}, handle, indent=2)
    except OSError:
        pass  # a session we can't save is not worth interrupting the user for


# =============================================================================
# SECTION 2: CALIBRATION -- FINDING THE A4 SHEET
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


def calibrate_photo(image_path):
    """
    Runs paper detection on a saved photo.

    Returns (pixels_per_cm or None, reason or None, candidates). The
    candidate boxes come back rather than being drawn here: the GUI
    overlays them on its own canvas so the user can see what the detector
    considered and why it rejected each one.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, f"Could not load image at {image_path}", []

    pixels_per_cm, reason, _, candidates = detect_paper_and_get_scale(
        image, return_all_candidates=True)
    return pixels_per_cm, reason, candidates


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


def _ensure_model_downloaded(path, url, label):
    if not os.path.exists(path):
        print(f"Downloading {label} model (one-time)...")
        urllib.request.urlretrieve(url, path)
        print("Model downloaded.")


def _ensure_pose_model_downloaded():
    _ensure_model_downloaded(POSE_MODEL_PATH, POSE_MODEL_URL, "pose landmarker")


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
    """Either path may be None (that photo was skipped) -- we just omit its results."""
    results = {}

    if arms_photo_path:
        landmarks, _, _ = detect_pose_landmarks(arms_photo_path)
        if landmarks:
            for side in ["left", "right"]:
                results[f"{side}_upper_arm_cm"] = measure_segment(
                    landmarks, f"{side}_shoulder", f"{side}_elbow", pixels_per_cm, "upper_arm")
                results[f"{side}_forearm_cm"] = measure_segment(
                    landmarks, f"{side}_elbow", f"{side}_wrist", pixels_per_cm, "forearm")

    if legs_photo_path:
        landmarks, _, _ = detect_pose_landmarks(legs_photo_path)
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
# SECTION 3a: SHARED HELPERS FOR THE HAND/FACE DETECTORS
# =============================================================================
# The hand and face landmarkers expect their subject to fill a decent part
# of the frame. On a 4000px full-body photo the face is only a couple of
# hundred pixels wide and both detectors simply return nothing.
#
# Rather than demand a separate close-up, we fall back to cropping: the
# POSE landmarker happily finds a whole person at that scale, so we use its
# eye/wrist landmarks to cut out the region of interest and re-run the
# detector on that crop.
#
# The crop is a plain slice with NO resizing, so pixels-per-cm is identical
# in the crop and the original -- distances measured in the crop convert to
# centimetres with the same calibration scale.


def _load_bgr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image at {image_path}")
    return image


def _crop_region(image, center_xy, half_width, half_height):
    """Returns the cropped sub-image, or None if the region is degenerate."""
    image_height, image_width = image.shape[:2]
    center_x, center_y = center_xy
    x1 = max(0, int(center_x - half_width))
    x2 = min(image_width, int(center_x + half_width))
    y1 = max(0, int(center_y - half_height))
    y2 = min(image_height, int(center_y + half_height))
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    return image[y1:y2, x1:x2]


def _to_mp_image(bgr_image):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def _visible(landmarks_dict, *names):
    """True only if every named landmark exists and clears MIN_VISIBILITY."""
    for name in names:
        landmark = landmarks_dict.get(name)
        if not landmark or landmark["visibility"] < MIN_VISIBILITY:
            return False
    return True


def _head_crop_from_pose(image, image_path):
    """
    Cuts out the head using the pose landmarker's eye positions. The crop
    is sized in multiples of the inter-eye distance: interocular distance
    is roughly 6.3cm and head width roughly 15cm, so a half-width of 2.2x
    interocular leaves comfortable margin for hair and a turned head.
    """
    landmarks, _, _ = detect_pose_landmarks(image_path)
    if not landmarks or not _visible(landmarks, "left_eye", "right_eye"):
        return None

    left_eye = landmarks["left_eye"]["pixel_coords"]
    right_eye = landmarks["right_eye"]["pixel_coords"]
    interocular_px = euclidean_distance(left_eye, right_eye)
    if interocular_px < 5:
        return None

    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = (left_eye[1] + right_eye[1]) / 2
    # Eyes sit below the middle of the head, so bias the crop upward.
    center_y -= 0.4 * interocular_px
    return _crop_region(image, (center_x, center_y),
                        2.2 * interocular_px, 2.8 * interocular_px)


def _hand_crop_from_pose(image, image_path):
    """
    Cuts out the hand using the pose landmarker's elbow and wrist. The
    forearm gives us a length scale in the same photo; the hand extends
    beyond the wrist along the elbow->wrist direction.
    """
    landmarks, _, _ = detect_pose_landmarks(image_path)
    if not landmarks:
        return None

    for side in ["left", "right"]:
        if not _visible(landmarks, f"{side}_elbow", f"{side}_wrist"):
            continue
        elbow = landmarks[f"{side}_elbow"]["pixel_coords"]
        wrist = landmarks[f"{side}_wrist"]["pixel_coords"]
        forearm_px = euclidean_distance(elbow, wrist)
        if forearm_px < 20:
            continue

        # Step half a forearm past the wrist to centre on the hand itself.
        direction_x = (wrist[0] - elbow[0]) / forearm_px
        direction_y = (wrist[1] - elbow[1]) / forearm_px
        center = (wrist[0] + 0.5 * forearm_px * direction_x,
                  wrist[1] + 0.5 * forearm_px * direction_y)

        crop = _crop_region(image, center, 0.9 * forearm_px, 0.9 * forearm_px)
        if crop is not None:
            return crop
    return None


# =============================================================================
# SECTION 3b: HAND LENGTH (MediaPipe Hand Landmarker)
# =============================================================================

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe hand landmark indices we use.
HAND_WRIST = 0
HAND_INDEX_MCP = 5
HAND_MIDDLE_TIP = 12
HAND_PINKY_MCP = 17

# No correction factor is applied to hand measurements. The limb factors in
# CORRECTION_FACTORS came from tape-measuring one subject's arms and legs --
# no equivalent ground truth exists for hands, so inventing a multiplier
# here would repeat the mistake the height comment above documents.


def measure_hand_from_photo(hands_photo_path, pixels_per_cm):
    """
    Returns {"hand_length_cm", "hand_breadth_cm"} or None.

    hand_length  = wrist (landmark 0) to middle fingertip (landmark 12).
                   This is the measurement glove charts key off.
    hand_breadth = across the knuckles, index MCP (5) to pinky MCP (17).

    Accuracy caveat: the palm must be flat-on to the camera and in the same
    plane as the calibration sheet. A tilted or cupped hand foreshortens
    both numbers, and unlike the pose landmarks there is no visibility
    score to catch it.
    """
    _ensure_model_downloaded(HAND_MODEL_PATH, HAND_MODEL_URL, "hand landmarker")

    bgr_image = _load_bgr(hands_photo_path)
    if bgr_image is None:
        return None

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
    )

    def detect(image):
        with HandLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(_to_mp_image(image))
        return result.hand_landmarks[0] if result.hand_landmarks else None

    target = bgr_image
    points = detect(target)
    if points is None:
        # Probably a full-body shot -- retry on a pose-guided crop.
        crop = _hand_crop_from_pose(bgr_image, hands_photo_path)
        if crop is not None:
            target = crop
            points = detect(target)
            if points is not None:
                print("Hand found by cropping around the wrist "
                      "(a close-up of the palm would measure more reliably).")
    if points is None:
        print(f"No hand detected in {hands_photo_path}")
        return None

    image_height, image_width = target.shape[:2]

    def px(index):
        return _get_pixel_coords(points[index], image_width, image_height)

    length_px = euclidean_distance(px(HAND_WRIST), px(HAND_MIDDLE_TIP))
    breadth_px = euclidean_distance(px(HAND_INDEX_MCP), px(HAND_PINKY_MCP))

    return {
        "hand_length_cm": length_px / pixels_per_cm,
        "hand_breadth_cm": breadth_px / pixels_per_cm,
    }


# =============================================================================
# SECTION 3c: HEAD CIRCUMFERENCE (MediaPipe Face Landmarker)
# =============================================================================

FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Canonical FaceMesh face-oval contour indices. We take the widest
# horizontal span across the whole contour rather than trusting one
# landmark pair, so a slightly off-centre face still gives a sane width.
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]
FACE_LEFT_EDGE = 234    # face oval, subject's right cheek
FACE_RIGHT_EDGE = 454   # face oval, subject's left cheek
NOSE_TIP = 1

# --- Head circumference model, and why it is only an estimate ---
# A helmet is sized by a tape wrapped around the skull. A frontal photo
# gives us exactly one of the two axes that loop needs: the width. So we
# model the head as an ellipse and derive the missing front-to-back axis
# from published anthropometry:
#
#   1. FACE_WIDTH_TO_HEAD_BREADTH -- the face-oval contour tracks the
#      cheeks/jaw, which sit slightly inside the widest part of the skull
#      (and inside any hair). Scale up modestly to reach skull breadth.
#   2. CEPHALIC_INDEX -- (head breadth / head length) * 100. Adult values
#      typically land in the 75-85 range; 80 is the mid-point we assume.
#      head_length = head_breadth / (CEPHALIC_INDEX / 100)
#   3. Ramanujan's ellipse-perimeter approximation over those two axes.
#
# Sanity check against population averages: a typical adult head breadth
# of 15.5cm and length of 19.4cm gives 55.0cm by this model, against a
# real adult mean of roughly 55-57cm. So the model is in the right place
# on average -- but an individual whose cephalic index is 75 or 85 rather
# than 80 will be off by around 2cm, which is a whole helmet size. Treat
# the output as a starting point and confirm with a tape measure.
FACE_WIDTH_TO_HEAD_BREADTH = 1.06
CEPHALIC_INDEX = 80.0
HEAD_YAW_TOLERANCE = 0.18  # max nose offset from centre, as a fraction of face width


def _ellipse_perimeter(semi_axis_a, semi_axis_b):
    """Ramanujan's second approximation -- accurate to <1e-5 for head-like ratios."""
    a, b = semi_axis_a, semi_axis_b
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def measure_head_from_photo(head_photo_path, pixels_per_cm):
    """
    Returns {"head_width_cm", "head_breadth_cm", "head_circumference_cm",
             "warning"} or None. See the model notes above -- the
    circumference is a modelled estimate, not a measurement.
    """
    _ensure_model_downloaded(FACE_MODEL_PATH, FACE_MODEL_URL, "face landmarker")

    bgr_image = _load_bgr(head_photo_path)
    if bgr_image is None:
        return None

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )

    def detect(image):
        with FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(_to_mp_image(image))
        return result.face_landmarks[0] if result.face_landmarks else None

    target = bgr_image
    points = detect(target)
    cropped = False
    if points is None:
        # Probably a full-body shot -- retry on a pose-guided crop.
        crop = _head_crop_from_pose(bgr_image, head_photo_path)
        if crop is not None:
            target = crop
            points = detect(target)
            cropped = points is not None
    if points is None:
        print(f"No face detected in {head_photo_path}")
        return None

    image_height, image_width = target.shape[:2]

    oval_xs = [points[i].x * image_width for i in FACE_OVAL_INDICES]
    face_width_px = max(oval_xs) - min(oval_xs)
    if face_width_px <= 0:
        return None

    # Yaw guard: on a straight-on face the nose sits near the midpoint of
    # the cheek landmarks. A large offset means the head is turned, which
    # foreshortens the width and silently shrinks the whole estimate.
    warnings = []
    left_x = points[FACE_LEFT_EDGE].x * image_width
    right_x = points[FACE_RIGHT_EDGE].x * image_width
    nose_offset = abs((points[NOSE_TIP].x * image_width) - (left_x + right_x) / 2)
    if nose_offset / face_width_px > HEAD_YAW_TOLERANCE:
        warnings.append("head looks turned away from the camera -- circumference is "
                        "likely underestimated; retake facing straight on")
    if cropped:
        warnings.append(f"face was only {face_width_px:.0f}px wide and had to be found by "
                        "cropping a wider shot -- a head close-up would be far more accurate")
    warning = "; ".join(warnings) if warnings else None

    face_width_cm = face_width_px / pixels_per_cm
    head_breadth_cm = face_width_cm * FACE_WIDTH_TO_HEAD_BREADTH
    head_length_cm = head_breadth_cm / (CEPHALIC_INDEX / 100.0)
    circumference_cm = _ellipse_perimeter(head_length_cm / 2, head_breadth_cm / 2)

    return {
        "head_width_cm": face_width_cm,
        "head_breadth_cm": head_breadth_cm,
        "head_circumference_cm": circumference_cm,
        "warning": warning,
    }


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
#  - Batting gloves: hand length (wrist-to-fingertip). Now measured via
#    the hand landmarker, but no numeric published cricket-glove chart was
#    available to key off -- banding below is generic, flagged as such.
#  - Helmets: head circumference. Now ESTIMATED (not measured) from face
#    width via the ellipse model in Section 3c. Banding below is the
#    common junior/small/medium/large split used across cricket helmet
#    brands; individual brands differ by a centimetre or two.

NOT_YET_AVAILABLE = "Not available yet -- needs a measurement this pipeline doesn't capture yet"

VERIFY_IN_PERSON = "verify in person before buying"


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


def recommend_batting_glove_size(hand_length_cm):
    """
    Batting gloves are sized by hand length (wrist crease to middle
    fingertip). As with arm guards, no numeric cm chart from a named
    cricket brand was available to key off, so these bands are the
    generic youth/adult splits used across glove sizing generally --
    a starting point, not a verified cricket-specific chart.
    """
    if hand_length_cm < 15.5:
        return f"Junior / Boys (rough banding -- no published cricket chart found, {VERIFY_IN_PERSON})"
    elif hand_length_cm < 17.5:
        return f"Youth (rough banding -- no published cricket chart found, {VERIFY_IN_PERSON})"
    elif hand_length_cm < 19.0:
        return f"Small Adult / Mens Small (rough banding -- no published cricket chart found, {VERIFY_IN_PERSON})"
    elif hand_length_cm < 20.5:
        return f"Adult / Mens (rough banding -- no published cricket chart found, {VERIFY_IN_PERSON})"
    else:
        return f"Large Adult (rough banding -- no published cricket chart found, {VERIFY_IN_PERSON})"


def recommend_helmet_size(head_circumference_cm):
    """
    Helmets are sized by head circumference. Our circumference is a
    MODELLED ESTIMATE from a frontal photo (see Section 3c), not a tape
    measurement -- an individual's skull proportions can move it by a
    couple of centimetres, which is a whole size. Always tape-measure
    before buying a helmet.
    """
    if head_circumference_cm < 52:
        return f"Junior / Small (approx <52cm -- {VERIFY_IN_PERSON})"
    elif head_circumference_cm < 55:
        return f"Small (approx 52-55cm -- {VERIFY_IN_PERSON})"
    elif head_circumference_cm < 58:
        return f"Medium (approx 55-58cm -- {VERIFY_IN_PERSON})"
    elif head_circumference_cm < 61:
        return f"Large (approx 58-61cm -- {VERIFY_IN_PERSON})"
    else:
        return f"Extra Large (approx 61cm+ -- {VERIFY_IN_PERSON})"


def recommend_gear(measurements, height_cm=None, hand=None, head=None):
    """
    Takes the measurements dict from measure_arms_and_legs() (plus
    optionally a separately-computed height_cm, the hand dict from
    measure_hand_from_photo() and the head dict from
    measure_head_from_photo()) and returns gear recommendations.

    Uses LEFT side measurements as representative (left/right should be
    close -- see the symmetry check built earlier).
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

    if hand and hand.get("hand_length_cm"):
        recommendations["batting_gloves"] = recommend_batting_glove_size(hand["hand_length_cm"])
    else:
        recommendations["batting_gloves"] = "Could not measure hand length"

    if head and head.get("head_circumference_cm"):
        helmet = recommend_helmet_size(head["head_circumference_cm"])
        if head.get("warning"):
            helmet += f"  [!] {head['warning']}"
        recommendations["helmet"] = helmet
    else:
        recommendations["helmet"] = "Could not measure head"

    return recommendations


# =============================================================================
# SECTION 5: MANUAL MARKING -- THE FALLBACK WHEN DETECTION IS WRONG
# =============================================================================
# Every automatic step here can fail quietly rather than loudly: the paper
# detector can lock onto a bright wall tile, and the pose landmarker will
# happily return a confident elbow that is 30px off. When that happens the
# user can see the right answer perfectly well -- they just need a way to
# tell us. So: click the points yourself.
#
# Click coordinates are captured on a screen-sized copy of the photo and
# divided back through the scale factor, so the points returned are always
# in ORIGINAL image pixels and stay compatible with pixels_per_cm.
#
# --- Why manual measurements get NO correction factor ---
# CORRECTION_FACTORS exists because MediaPipe's joint landmarks sit at
# estimated joint centres and came out consistently short against a tape
# measure. Those multipliers were fitted to automatic landmarks. A human
# clicking a knee has a completely different error profile, and scaling a
# hand-placed point by 1.34x would simply inflate it. We cannot separate
# how much of the original gap was landmark placement versus lens/plane
# error, so we do not pretend to: manual numbers are reported raw. If a
# manual result and an automatic one disagree, that gap is information --
# it is the correction factor being tested, and worth recording.


def _order_quad(points):
    """
    Sorts four clicked corners into cyclic order -- going around the edge
    of the sheet -- so the user can click them in any order or direction.

    This sorts by angle about the centroid rather than using the common
    "smallest x+y is top-left" trick. That trick degenerates when the
    sheet sits near 45 degrees: one corner wins two roles, an edge
    silently collapses to zero, and the scale comes out wrong with no
    warning. Angular sort is correct for any convex quad at any rotation,
    and cyclic order is all the edge lengths need.
    """
    pts = np.array(points, dtype=np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)]


def scale_from_paper_corners(corners):
    """
    Four clicked A4 corners -> (pixels_per_cm, warning or None).

    Both edge pairs are used, not just one: opposite edges of a tilted
    sheet differ under perspective, and the size of that disagreement is
    a useful warning that the sheet wasn't flat-on to the camera.
    """
    ordered = _order_quad(corners)
    edges = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i])) for i in range(4)]
    # Opposite edges are two apart in cyclic order.
    edge_pair_one = (edges[0] + edges[2]) / 2
    edge_pair_two = (edges[1] + edges[3]) / 2

    if min(edge_pair_one, edge_pair_two) < 1:
        return None, "Those corners are too close together to measure."

    # The sheet may be held either way up; the longer side is always 29.7cm.
    long_px = max(edge_pair_one, edge_pair_two)
    short_px = min(edge_pair_one, edge_pair_two)

    scale_from_long = long_px / PAPER_REAL_HEIGHT_CM
    scale_from_short = short_px / PAPER_REAL_WIDTH_CM
    pixels_per_cm = (scale_from_long + scale_from_short) / 2

    warning = None
    disagreement = abs(scale_from_long - scale_from_short) / pixels_per_cm
    if disagreement > 0.10:
        warning = (f"The two sides disagree by {disagreement * 100:.0f}% "
                   f"({scale_from_short:.2f} vs {scale_from_long:.2f} px/cm). The sheet was "
                   "probably tilted rather than flat-on -- every measurement inherits this error.")
    return pixels_per_cm, warning


def _distance_cm(points, index_a, index_b, pixels_per_cm):
    return euclidean_distance(points[index_a], points[index_b]) / pixels_per_cm


def arm_from_points(points, side, pixels_per_cm):
    """Shoulder, elbow, wrist -> upper arm and forearm lengths."""
    return {
        f"{side}_upper_arm_cm": _distance_cm(points, 0, 1, pixels_per_cm),
        f"{side}_forearm_cm": _distance_cm(points, 1, 2, pixels_per_cm),
    }


def leg_from_points(points, side, pixels_per_cm):
    """Hip, knee, ankle -> thigh and shin lengths."""
    return {
        f"{side}_thigh_cm": _distance_cm(points, 0, 1, pixels_per_cm),
        f"{side}_shin_cm": _distance_cm(points, 1, 2, pixels_per_cm),
    }


def height_from_points(points, pixels_per_cm):
    """
    Top of head, ground at heels -> height in cm.

    This is the one manual measurement that is strictly better than its
    automatic counterpart rather than just a fallback: the pose landmarker
    has no crown landmark and has to add HEAD_TOP_OFFSET_CM, a population
    average. Clicking the actual top of the head removes that assumption.
    """
    return _distance_cm(points, 0, 1, pixels_per_cm)


def hand_from_points(points, pixels_per_cm):
    """Wrist crease, middle fingertip, index knuckle, little-finger knuckle."""
    return {
        "hand_length_cm": _distance_cm(points, 0, 1, pixels_per_cm),
        "hand_breadth_cm": _distance_cm(points, 2, 3, pixels_per_cm),
    }


def head_from_points(points, pixels_per_cm):
    """
    Widest point each side of the head -> the same ellipse model the
    automatic path uses, with one difference: the face landmarker traces
    the cheeks and jaw, so its width is scaled up by
    FACE_WIDTH_TO_HEAD_BREADTH to reach the skull. A human clicking the
    widest visible points is already marking skull breadth including
    hair, so no such scaling is applied here.
    """
    head_breadth_cm = _distance_cm(points, 0, 1, pixels_per_cm)
    head_length_cm = head_breadth_cm / (CEPHALIC_INDEX / 100.0)
    return {
        "head_width_cm": head_breadth_cm,
        "head_breadth_cm": head_breadth_cm,
        "head_circumference_cm": _ellipse_perimeter(head_length_cm / 2, head_breadth_cm / 2),
        "warning": "marked by hand; circumference still assumes a "
                   f"cephalic index of {CEPHALIC_INDEX:.0f} -- tape-measure to confirm",
    }


# What to ask the user to click, per step. Steps flagged `sided` have
# their prompts filled in with "left" or "right" before being shown.
MANUAL_MARKS = {
    "calibration": {
        "prompts": ["a corner of the A4 sheet",
                    "the next corner (going around the edge)",
                    "the third corner",
                    "the last corner"],
        "closed": True,
        "sided": False,
    },
    "arms": {
        "prompts": ["the {side} SHOULDER joint",
                    "the {side} ELBOW joint",
                    "the {side} WRIST"],
        "closed": False,
        "sided": True,
    },
    "legs": {
        "prompts": ["the {side} HIP joint",
                    "the {side} KNEE (centre of the kneecap)",
                    "the {side} ANKLE bone"],
        "closed": False,
        "sided": True,
    },
    "full_body_height": {
        "prompts": ["the very TOP of the head (including hair)",
                    "the GROUND at the base of the heels"],
        "closed": False,
        "sided": False,
    },
    "hands": {
        "prompts": ["the WRIST CREASE (where the palm meets the wrist)",
                    "the tip of the MIDDLE FINGER",
                    "the INDEX finger knuckle",
                    "the LITTLE finger knuckle"],
        "closed": False,
        "sided": False,
    },
    "head": {
        "prompts": ["the widest point on the LEFT side of the head",
                    "the widest point on the RIGHT side of the head"],
        "closed": False,
        "sided": False,
    },
}


def manual_prompts(step_key, side="left"):
    return [p.format(side=side) for p in MANUAL_MARKS[step_key]["prompts"]]


def apply_manual_marks(step_key, points, side, pixels_per_cm, results):
    """
    Folds a set of clicked points into `results`, in the same shape the
    automatic path produces. Returns a short description of what changed.
    """
    if step_key == "arms":
        results["segments"].update(arm_from_points(points, side, pixels_per_cm))
        return f"{side} arm"
    if step_key == "legs":
        results["segments"].update(leg_from_points(points, side, pixels_per_cm))
        return f"{side} leg"
    if step_key == "full_body_height":
        results["height_cm"] = height_from_points(points, pixels_per_cm)
        return "height"
    if step_key == "hands":
        results["hand"] = hand_from_points(points, pixels_per_cm)
        return "hand"
    if step_key == "head":
        results["head"] = head_from_points(points, pixels_per_cm)
        return "head"
    raise ValueError(f"no manual handler for {step_key}")


# =============================================================================
# SECTION 6: MEASUREMENT ORCHESTRATION
# =============================================================================
# Small, UI-agnostic layer between the detectors and the interface: run
# one step, file the answer, describe it. The GUI drives these from a
# worker thread, so nothing here may touch a widget.


def empty_results():
    return {"segments": {}, "height_cm": None, "hand": None, "head": None,
            "manual": set()}


def measure_step(step_key, photo_path, pixels_per_cm):
    """
    Runs the automatic measurement for one step. Returns a value in the
    same shape apply_manual_marks() produces, or None if nothing could be
    measured.
    """
    if step_key == "arms":
        return measure_arms_and_legs(photo_path, None, pixels_per_cm) or None
    if step_key == "legs":
        return measure_arms_and_legs(None, photo_path, pixels_per_cm) or None
    if step_key == "full_body_height":
        return measure_height_from_photo(photo_path, pixels_per_cm)
    if step_key == "hands":
        return measure_hand_from_photo(photo_path, pixels_per_cm)
    if step_key == "head":
        return measure_head_from_photo(photo_path, pixels_per_cm)
    raise ValueError(f"no automatic handler for {step_key}")


def store_step_result(step_key, value, results):
    """Files whatever measure_step() returned into the results dict."""
    if value is None:
        return
    if step_key in ("arms", "legs"):
        results["segments"].update(value)
    elif step_key == "full_body_height":
        results["height_cm"] = value
    elif step_key == "hands":
        results["hand"] = value
    elif step_key == "head":
        results["head"] = value


def summarise_step(step_key, results):
    """One-line human summary of what a step currently measures, or None."""
    segments = results["segments"]

    def sided(first_suffix, second_suffix, first_label, second_label):
        parts = []
        for side in ("left", "right"):
            first = segments.get(f"{side}_{first_suffix}")
            second = segments.get(f"{side}_{second_suffix}")
            if first is not None:
                parts.append(f"{side} {first_label} {first:.1f}cm")
            if second is not None:
                parts.append(f"{side} {second_label} {second:.1f}cm")
        return ", ".join(parts) or None

    if step_key == "arms":
        return sided("upper_arm_cm", "forearm_cm", "upper arm", "forearm")
    if step_key == "legs":
        return sided("thigh_cm", "shin_cm", "thigh", "shin")
    if step_key == "full_body_height":
        height = results["height_cm"]
        return f"height {height:.1f}cm" if height else None
    if step_key == "hands":
        hand = results["hand"]
        if not hand:
            return None
        return f"length {hand['hand_length_cm']:.1f}cm, breadth {hand['hand_breadth_cm']:.1f}cm"
    if step_key == "head":
        head = results["head"]
        if not head:
            return None
        return (f"width {head['head_width_cm']:.1f}cm, "
                f"circumference ~{head['head_circumference_cm']:.1f}cm")
    return None


def measurement_rows(results):
    """(name, value, source) rows for the results table."""
    manual = results["manual"]
    rows = []

    for key, value in results["segments"].items():
        group = "arms" if "arm" in key else "legs"
        rows.append((key,
                     f"{value:.1f} cm" if value is not None else "could not measure",
                     "marked by hand" if group in manual else "automatic"))

    height = results["height_cm"]
    if height:
        source = ("marked by hand" if "full_body_height" in manual
                  else f"automatic (+{HEAD_TOP_OFFSET_CM:.0f}cm eye-to-crown average)")
        rows.append(("height_cm", f"{height:.1f} cm", source))

    hand = results["hand"]
    if hand:
        source = "marked by hand" if "hands" in manual else "automatic"
        rows.append(("hand_length_cm", f"{hand['hand_length_cm']:.1f} cm", source))
        rows.append(("hand_breadth_cm", f"{hand['hand_breadth_cm']:.1f} cm", source))

    head = results["head"]
    if head:
        source = "marked by hand" if "head" in manual else "automatic"
        rows.append(("head_width_cm", f"{head['head_width_cm']:.1f} cm", source))
        rows.append(("head_circumference_cm",
                     f"~{head['head_circumference_cm']:.1f} cm",
                     f"estimated from width (cephalic index {CEPHALIC_INDEX:.0f})"))
    return rows


# =============================================================================
# SECTION 7: GRAPHICAL INTERFACE (PySide6)
# =============================================================================
# Everything above this line is UI-agnostic: it takes paths and numbers and
# returns numbers. This section is the only part that knows about widgets.
#
# Two rules keep that separation honest:
#   1. Nothing below calls print() for anything the user needs to see --
#      it goes to a label, a status bar or a dialog.
#   2. Nothing above is called from the paint or event loop directly if it
#      touches MediaPipe. Detection takes a second or more per photo, and
#      running it inline would freeze the window mid-click. It runs in a
#      Worker thread instead.


class Worker(QtCore.QThread):
    """Runs one callable off the UI thread and hands back its return value."""

    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self._function = function

    def run(self):
        try:
            self.done.emit(self._function())
        except Exception as exc:  # a detector blowing up must not kill the app
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def bgr_to_qimage(bgr_image):
    """OpenCV BGR array -> QImage that owns its own pixels."""
    height, width = bgr_image.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    image = QtGui.QImage(rgb.data, width, height, 3 * width,
                         QtGui.QImage.Format.Format_RGB888)
    # copy() detaches from the numpy buffer, which is about to go out of scope.
    return image.copy()


class ImageCanvas(QtWidgets.QWidget):
    """
    Shows a photo (or a live camera frame) and, when asked, collects
    clicked points on top of it.

    The photo is letterboxed to fit the widget, so every click has to be
    mapped back through that transform -- `points()` always returns
    ORIGINAL image pixels, which is what pixels_per_cm is expressed in.
    A magnifier follows the cursor sampling the full-resolution image,
    because a 4000px photo shown in a 700px widget would otherwise round
    every click to the nearest six pixels of real image.
    """

    point_added = QtCore.Signal(int)
    marking_ready = QtCore.Signal()

    LOUPE_SIZE = 170
    LOUPE_ZOOM = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self._image = None          # numpy BGR, original resolution
        self._pixmap = None
        self._placeholder = "No photo yet"
        self._overlay_text = None   # e.g. the capture countdown
        self._boxes = []            # calibration candidates: (rect, passed)

        self._marking = False
        self._prompts = []
        self._closed_shape = False
        self._points = []           # image coordinates
        self._cursor = None         # widget coordinates

    # -- content -----------------------------------------------------------

    def set_image(self, bgr_image):
        self._image = bgr_image
        self._pixmap = QtGui.QPixmap.fromImage(bgr_to_qimage(bgr_image)) \
            if bgr_image is not None else None
        self.update()

    def set_placeholder(self, text):
        self._placeholder = text
        self.update()

    def clear(self):
        self._image = None
        self._pixmap = None
        self._boxes = []
        self.cancel_marking()
        self.update()

    def set_overlay_text(self, text):
        self._overlay_text = text
        self.update()

    def set_boxes(self, boxes):
        """boxes: list of ((x, y, w, h), passed_bool) in image coordinates."""
        self._boxes = boxes
        self.update()

    # -- marking -----------------------------------------------------------

    def begin_marking(self, prompts, closed_shape=False):
        self._marking = True
        self._prompts = list(prompts)
        self._closed_shape = closed_shape
        self._points = []
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.update()

    def cancel_marking(self):
        self._marking = False
        self._points = []
        self._prompts = []
        self.unsetCursor()
        self.update()

    def undo_point(self):
        if self._points:
            self._points.pop()
            self.point_added.emit(len(self._points))
            self.update()

    def reset_points(self):
        self._points = []
        self.point_added.emit(0)
        self.update()

    def points(self):
        return list(self._points)

    def is_complete(self):
        return self._marking and len(self._points) == len(self._prompts)

    # -- geometry ----------------------------------------------------------

    def _transform(self):
        """Returns (scale, offset_x, offset_y) mapping image -> widget."""
        if self._pixmap is None:
            return None
        image_width = self._pixmap.width()
        image_height = self._pixmap.height()
        scale = min(self.width() / image_width, self.height() / image_height)
        return (scale,
                (self.width() - image_width * scale) / 2,
                (self.height() - image_height * scale) / 2)

    def _widget_to_image(self, position):
        transform = self._transform()
        if transform is None:
            return None
        scale, offset_x, offset_y = transform
        x = (position.x() - offset_x) / scale
        y = (position.y() - offset_y) / scale
        if not (0 <= x < self._pixmap.width() and 0 <= y < self._pixmap.height()):
            return None
        return (x, y)

    def _image_to_widget(self, point):
        scale, offset_x, offset_y = self._transform()
        return QtCore.QPointF(point[0] * scale + offset_x, point[1] * scale + offset_y)

    # -- events ------------------------------------------------------------

    def mouseMoveEvent(self, event):
        self._cursor = event.position()
        if self._marking:
            self.update()

    def leaveEvent(self, event):
        self._cursor = None
        self.update()

    def mousePressEvent(self, event):
        if not self._marking or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if len(self._points) >= len(self._prompts):
            return
        image_point = self._widget_to_image(event.position())
        if image_point is None:
            return
        self._points.append(image_point)
        self.point_added.emit(len(self._points))
        if self.is_complete():
            self.marking_ready.emit()
        self.update()

    def keyPressEvent(self, event):
        if not self._marking:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == QtCore.Qt.Key.Key_U:
            self.undo_point()
        elif key == QtCore.Qt.Key.Key_R:
            self.reset_points()
        else:
            super().keyPressEvent(event)

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#1b1b1f"))

        if self._pixmap is None:
            painter.setPen(QtGui.QColor("#8a8a92"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                             self._placeholder)
            return

        scale, offset_x, offset_y = self._transform()
        painter.drawPixmap(
            QtCore.QRectF(offset_x, offset_y,
                          self._pixmap.width() * scale, self._pixmap.height() * scale),
            self._pixmap, QtCore.QRectF(self._pixmap.rect()))

        self._paint_boxes(painter, scale, offset_x, offset_y)
        if self._marking:
            self._paint_marks(painter)
            self._paint_banner(painter)
            self._paint_loupe(painter)
        if self._overlay_text:
            self._paint_overlay_text(painter)

    def _paint_boxes(self, painter, scale, offset_x, offset_y):
        for (x, y, w, h), passed in self._boxes:
            colour = QtGui.QColor("#4ade80") if passed else QtGui.QColor("#f87171")
            painter.setPen(QtGui.QPen(colour, 2))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(QtCore.QRectF(x * scale + offset_x, y * scale + offset_y,
                                           w * scale, h * scale))

    def _paint_marks(self, painter):
        if len(self._points) > 1:
            painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 2))
            path = [self._image_to_widget(p) for p in self._points]
            for a, b in zip(path, path[1:]):
                painter.drawLine(a, b)
            if self._closed_shape and self.is_complete():
                painter.drawLine(path[-1], path[0])

        for index, point in enumerate(self._points):
            centre = self._image_to_widget(point)
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            painter.setBrush(QtGui.QColor("#ef4444"))
            painter.drawEllipse(centre, 6, 6)
            painter.setPen(QtGui.QColor("#ffffff"))
            painter.drawText(centre + QtCore.QPointF(11, -9), str(index + 1))

    def _paint_banner(self, painter):
        if self.is_complete():
            text = "All points placed -- press Accept (or Undo / Reset)"
        else:
            text = (f"Click point {len(self._points) + 1} of {len(self._prompts)}: "
                    f"{self._prompts[len(self._points)]}")
        rect = QtCore.QRectF(0, 0, self.width(), 34)
        painter.fillRect(rect, QtGui.QColor(0, 0, 0, 165))
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(rect.adjusted(12, 0, -12, 0),
                         QtCore.Qt.AlignmentFlag.AlignVCenter, text)

    def _paint_loupe(self, painter):
        if self._cursor is None or self._image is None:
            return
        image_point = self._widget_to_image(self._cursor)
        if image_point is None:
            return

        size = self.LOUPE_SIZE
        if min(self.width(), self.height()) < size + 40:
            return

        half = max(4, int(size / (2 * self.LOUPE_ZOOM)))
        height, width = self._image.shape[:2]
        x1, x2 = max(0, int(image_point[0]) - half), min(width, int(image_point[0]) + half)
        y1, y2 = max(0, int(image_point[1]) - half), min(height, int(image_point[1]) + half)
        patch = self._image[y1:y2, x1:x2]
        if patch.size == 0:
            return

        zoomed = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)
        pixmap = QtGui.QPixmap.fromImage(bgr_to_qimage(zoomed))

        # Park the loupe in the corner furthest from the cursor.
        margin = 12
        left = margin if self._cursor.x() > self.width() / 2 else self.width() - size - margin
        top = 46 if self._cursor.y() > self.height() / 2 else self.height() - size - margin
        painter.drawPixmap(int(left), int(top), pixmap)

        # drawRect fills with the current brush, and _paint_marks left it set
        # to the marker colour -- without this the loupe becomes a red block.
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        centre_x, centre_y = left + size / 2, top + size / 2
        painter.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 1))
        painter.drawLine(QtCore.QPointF(centre_x, top), QtCore.QPointF(centre_x, top + size))
        painter.drawLine(QtCore.QPointF(left, centre_y), QtCore.QPointF(left + size, centre_y))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        painter.drawRect(QtCore.QRectF(left, top, size, size))

    def _paint_overlay_text(self, painter):
        font = painter.font()
        font.setPointSize(64)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#f87171"))
        painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                         self._overlay_text)


class StepPage(QtWidgets.QWidget):
    """One photo step: choose a source, then measure it or mark it by hand."""

    CAMERA_POLL_MS = 30

    def __init__(self, step, window):
        super().__init__()
        self.step = step
        self.window = window
        self.key = step["key"]
        self.is_calibration = self.key == "calibration"

        self._capture = None
        self._camera_timer = QtCore.QTimer(self)
        self._camera_timer.timeout.connect(self._poll_camera)
        self._countdown_deadline = None
        self._worker = None

        title = QtWidgets.QLabel(step["label"])
        title_font = title.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)

        instruction = QtWidgets.QLabel(f"{step['instruction']}\nNeeded for: {step['unlocks']}")
        instruction.setWordWrap(True)
        instruction.setStyleSheet("color: #9ca3af;")

        self.canvas = ImageCanvas()
        self.canvas.set_placeholder("No photo yet -- use the camera or pick one from your gallery")
        self.canvas.point_added.connect(lambda _: self._sync_marking_buttons())
        self.canvas.marking_ready.connect(self._sync_marking_buttons)

        # Which file this step is actually reading. Worth showing: photos
        # are used where they live now, so "which one is loaded?" is a
        # real question the user can otherwise only guess at.
        self.source = QtWidgets.QLabel("")
        self.source.setStyleSheet("color: #6b7280;")
        self.source.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)

        self.buttons = QtWidgets.QStackedWidget()
        self.buttons.addWidget(self._build_idle_bar())
        self.buttons.addWidget(self._build_camera_bar())
        self.buttons.addWidget(self._build_marking_bar())
        self.buttons.setFixedHeight(46)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(instruction)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.source)
        layout.addWidget(self.status)
        layout.addWidget(self.buttons)

    # -- button bars -------------------------------------------------------

    def _build_idle_bar(self):
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)

        self.camera_button = QtWidgets.QPushButton("Use camera")
        self.camera_button.clicked.connect(self._start_camera)
        self.gallery_button = QtWidgets.QPushButton("Choose from gallery")
        self.gallery_button.clicked.connect(self._choose_from_gallery)
        self.mark_button = QtWidgets.QPushButton(
            "Mark corners by hand" if self.is_calibration else "Mark by hand")
        self.mark_button.clicked.connect(self._start_marking)
        self.remeasure_button = QtWidgets.QPushButton("Re-detect")
        self.remeasure_button.clicked.connect(self.measure)
        self.skip_button = QtWidgets.QPushButton("Skip")
        self.skip_button.clicked.connect(self._skip)

        for widget in (self.camera_button, self.gallery_button, self.mark_button,
                       self.remeasure_button):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.skip_button)
        return bar

    def _build_camera_bar(self):
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        capture = QtWidgets.QPushButton(f"Capture ({COUNTDOWN_SECONDS}s countdown)")
        capture.clicked.connect(self._start_countdown)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self._stop_camera)
        row.addWidget(capture)
        row.addWidget(cancel)
        row.addStretch(1)
        return bar

    def _build_marking_bar(self):
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)

        self.side_box = QtWidgets.QComboBox()
        self.side_box.addItems(["left", "right"])
        self.side_box.currentTextChanged.connect(self._restart_marking_for_side)
        self.side_label = QtWidgets.QLabel("Side:")
        sided = MANUAL_MARKS.get(self.key, {}).get("sided", False)
        self.side_label.setVisible(sided)
        self.side_box.setVisible(sided)

        self.undo_button = QtWidgets.QPushButton("Undo (u)")
        self.undo_button.clicked.connect(self.canvas.undo_point)
        self.reset_button = QtWidgets.QPushButton("Reset (r)")
        self.reset_button.clicked.connect(self.canvas.reset_points)
        self.accept_button = QtWidgets.QPushButton("Accept")
        self.accept_button.clicked.connect(self._accept_marks)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self._cancel_marking)

        row.addWidget(self.side_label)
        row.addWidget(self.side_box)
        row.addWidget(self.undo_button)
        row.addWidget(self.reset_button)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(self.accept_button)
        return bar

    # -- photo sources -----------------------------------------------------

    def _choose_from_gallery(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Choose a photo for: {self.step['label']}", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*)")
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            QtWidgets.QMessageBox.warning(self, "Unreadable file",
                                          "That file could not be read as an image.")
            return

        # Quality problems are a warning here, not a rejection: the camera
        # can always retake a shot, but an imported photo may be the only
        # one the user has.
        is_good, reason = check_photo_quality(image)
        if not is_good:
            answer = QtWidgets.QMessageBox.question(
                self, "This photo may not measure well",
                f"{reason}\n\nUse it anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes |
                QtWidgets.QMessageBox.StandardButton.No)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        # Used where it is. Nothing is copied, so choosing a photo can
        # never overwrite another one -- including the file you just chose.
        self._adopt_photo(path)

    def _start_camera(self):
        self._capture = cv2.VideoCapture(0)
        if not self._capture.isOpened():
            self._capture = None
            QtWidgets.QMessageBox.warning(self, "No camera",
                                          "Could not open a camera on this machine.")
            return
        self.buttons.setCurrentIndex(1)
        self.status.setText("Live preview -- press Capture when you are in position.")
        self._camera_timer.start(self.CAMERA_POLL_MS)

    def _stop_camera(self):
        self._camera_timer.stop()
        self._countdown_deadline = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.canvas.set_overlay_text(None)
        self.buttons.setCurrentIndex(0)
        self.refresh()

    def _start_countdown(self):
        self._countdown_deadline = time.time() + COUNTDOWN_SECONDS

    def _poll_camera(self):
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok:
            return
        self.canvas.set_image(frame)

        if self._countdown_deadline is None:
            return
        remaining = self._countdown_deadline - time.time()
        if remaining > 0:
            self.canvas.set_overlay_text(str(int(remaining) + 1))
            return

        self.canvas.set_overlay_text(None)
        self._countdown_deadline = None
        is_good, reason = check_photo_quality(frame)
        if not is_good:
            self.status.setText(f"Rejected: {reason}  --  adjust and capture again.")
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        destination = self._camera_destination()
        cv2.imwrite(destination, frame)
        self._stop_camera()
        self._adopt_photo(destination)

    def _camera_destination(self):
        """
        Where to write a capture: this step's own slot, unless another
        step is currently using that exact file (which it can be, since
        gallery photos are used in place and may point into OUTPUT_DIR).
        """
        destination = photo_path_for(self.key)
        in_use = [path for key, path in self.window.photos.items()
                  if key != self.key and same_file(path, destination)]
        if not in_use:
            return destination

        stem, extension = os.path.splitext(destination)
        index = 2
        while os.path.exists(f"{stem}_{index}{extension}"):
            index += 1
        return f"{stem}_{index}{extension}"

    def _skip(self):
        self.window.photos.pop(self.key, None)
        self.window.skipped.add(self.key)
        save_session_photos(self.window.photos)
        self.canvas.clear()
        self.canvas.set_placeholder("Skipped")
        self.source.setText("")
        self.status.setText(f"Skipped -- {self.step['unlocks']} will be unavailable.")
        self.window.step_changed(self.key)

    def _adopt_photo(self, path):
        self.window.photos[self.key] = path
        self.window.skipped.discard(self.key)
        save_session_photos(self.window.photos)
        self.refresh()
        self.measure()

    # -- manual marking ----------------------------------------------------

    def _start_marking(self):
        if self.key not in self.window.photos:
            QtWidgets.QMessageBox.information(
                self, "No photo", "Add a photo for this step first.")
            return
        if not self.is_calibration and not self.window.pixels_per_cm:
            QtWidgets.QMessageBox.information(
                self, "No scale yet",
                "Calibrate with the A4 sheet first -- without a scale, clicked "
                "points can't be converted to centimetres.")
            return
        self.buttons.setCurrentIndex(2)
        spec = MANUAL_MARKS[self.key]
        self.canvas.set_boxes([])
        self.canvas.begin_marking(manual_prompts(self.key, self.side_box.currentText()),
                                  spec["closed"])
        self._sync_marking_buttons()

    def _restart_marking_for_side(self):
        if self.buttons.currentIndex() == 2:
            self._start_marking()

    def _sync_marking_buttons(self):
        self.accept_button.setEnabled(self.canvas.is_complete())

    def _cancel_marking(self):
        self.canvas.cancel_marking()
        self.buttons.setCurrentIndex(0)
        self.refresh()

    def _accept_marks(self):
        points = self.canvas.points()
        self.canvas.cancel_marking()
        self.buttons.setCurrentIndex(0)

        if self.is_calibration:
            scale, warning = scale_from_paper_corners(points)
            if scale is None:
                QtWidgets.QMessageBox.warning(self, "Could not use those corners", warning)
                return
            self.window.set_scale(scale, source="marked by hand", warning=warning)
            self.refresh()
            return

        description = apply_manual_marks(self.key, points, self.side_box.currentText(),
                                         self.window.pixels_per_cm, self.window.results)
        self.window.results["manual"].add(self.key)
        self.status.setText(f"Updated {description} from your marks.")
        self.window.step_changed(self.key)
        self.refresh(keep_status=True)

    # -- measuring ---------------------------------------------------------

    def measure(self):
        path = self.window.photos.get(self.key)
        if not path:
            return
        if self.is_calibration:
            self._run(lambda: calibrate_photo(path), self._calibration_done,
                      "Looking for the A4 sheet...")
            return
        if not self.window.pixels_per_cm:
            self.status.setText("Waiting for calibration before this can be measured.")
            return
        scale = self.window.pixels_per_cm
        self._run(lambda: measure_step(self.key, path, scale), self._measure_done,
                  "Measuring...")

    def _run(self, function, on_done, busy_text):
        self.status.setText(busy_text)
        self._set_controls_enabled(False)
        self._worker = Worker(function, self)
        self._worker.done.connect(on_done)
        self._worker.failed.connect(self._worker_failed)
        self._worker.finished.connect(lambda: self._set_controls_enabled(True))
        self._worker.start()

    def _worker_failed(self, message):
        self.status.setText(f"Failed: {message}")

    def _set_controls_enabled(self, enabled):
        for button in (self.camera_button, self.gallery_button, self.mark_button,
                       self.remeasure_button, self.skip_button):
            button.setEnabled(enabled)

    def _calibration_done(self, outcome):
        pixels_per_cm, reason, candidates = outcome
        self.canvas.set_boxes([(c["box"], c["pixels_per_cm"] is not None)
                               for c in candidates if c["box"] is not None])
        if pixels_per_cm is None:
            self.status.setText(
                f"Could not find the A4 sheet: {reason}\n"
                "Green boxes passed, red ones were rejected. "
                "Use 'Mark corners by hand' to set the scale yourself.")
            self.window.set_scale(None)
            return
        self.window.set_scale(pixels_per_cm, source="detected automatically")
        # No keep_status: the "Looking for the A4 sheet..." message was
        # written before the worker started and would otherwise stick.
        self.refresh()

    def _measure_done(self, value):
        if value is None:
            self.status.setText(
                "Nothing could be measured in this photo. Try 'Mark by hand', "
                "or retake it with the whole body part clearly visible.")
            self.window.step_changed(self.key)
            return
        self.window.results["manual"].discard(self.key)
        store_step_result(self.key, value, self.window.results)
        self.window.step_changed(self.key)
        # No keep_status -- otherwise "Measuring..." stays on screen for
        # whichever page the user happens to be looking at.
        self.refresh()

    # -- display -----------------------------------------------------------

    def refresh(self, keep_status=False):
        path = self.window.photos.get(self.key)
        if path and os.path.exists(path):
            image = cv2.imread(path)
            if image is not None:
                self.canvas.set_image(image)
            self.source.setText(f"Using: {os.path.abspath(path)}")
        elif self.key in self.window.skipped:
            self.canvas.clear()
            self.canvas.set_placeholder("Skipped")
            self.source.setText("")
        else:
            self.canvas.clear()
            self.source.setText("")

        self.mark_button.setEnabled(bool(path))
        self.remeasure_button.setEnabled(bool(path))

        if keep_status:
            return
        self.status.setText(self.summary())

    def summary(self):
        if self.key in self.window.skipped:
            return f"Skipped -- {self.step['unlocks']} will be unavailable."
        if self.key not in self.window.photos:
            return "No photo yet."
        if self.is_calibration:
            if self.window.pixels_per_cm:
                return (f"Scale: {self.window.pixels_per_cm:.2f} px/cm "
                        f"({self.window.scale_source}).")
            return "Photo loaded, but no A4 sheet found yet."
        text = summarise_step(self.key, self.window.results)
        if not text:
            return "Photo loaded, not measured yet."
        if self.key in self.window.results["manual"]:
            return f"{text}   [marked by hand]"
        return text

    def state(self):
        """Sidebar status: 'done', 'attention', 'skipped' or 'empty'."""
        if self.key in self.window.skipped:
            return "skipped"
        if self.key not in self.window.photos:
            return "empty"
        if self.is_calibration:
            return "done" if self.window.pixels_per_cm else "attention"
        return "done" if summarise_step(self.key, self.window.results) else "attention"

    def closing(self):
        self._stop_camera()


class ResultsPage(QtWidgets.QWidget):
    """Measurements and the gear sizes derived from them."""

    def __init__(self, window):
        super().__init__()
        self.window = window

        title = QtWidgets.QLabel("Results")
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)

        self.scale_label = QtWidgets.QLabel()
        self.scale_label.setStyleSheet("color: #9ca3af;")

        self.measurements = self._make_table(["Measurement", "Value", "Source"])
        self.recommendations = self._make_table(["Item", "Recommended size"])

        self.caveat = QtWidgets.QLabel(
            "These are estimates from photos, not a fitting. Confirm sizes in person "
            "before buying -- especially the helmet, whose circumference is modelled "
            "from face width rather than measured.")
        self.caveat.setWordWrap(True)
        self.caveat.setStyleSheet("color: #fbbf24;")

        refresh = QtWidgets.QPushButton("Re-measure everything")
        refresh.clicked.connect(self.window.measure_all)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(refresh)
        row.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.scale_label)
        layout.addWidget(QtWidgets.QLabel("Measurements"))
        layout.addWidget(self.measurements, stretch=3)
        layout.addWidget(QtWidgets.QLabel("Cricket gear"))
        layout.addWidget(self.recommendations, stretch=2)
        layout.addWidget(self.caveat)
        layout.addLayout(row)

    @staticmethod
    def _make_table(headers):
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setStretchLastSection(True)
        table.setWordWrap(True)
        return table

    @staticmethod
    def _fill(table, rows):
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, text in enumerate(row):
                table.setItem(row_index, column_index,
                              QtWidgets.QTableWidgetItem(str(text)))

        # Size the label columns to their content, then let the final
        # column take the rest and wrap. Resizing every column to content
        # would size the last one to its longest unwrapped line, which
        # pushes the sizing notes off the edge behind a scrollbar.
        header = table.horizontalHeader()
        last = table.columnCount() - 1
        for column in range(last):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(last, QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.resizeRowsToContents()

    def refresh(self):
        if self.window.pixels_per_cm:
            self.scale_label.setText(
                f"Scale: {self.window.pixels_per_cm:.2f} px/cm ({self.window.scale_source})")
        else:
            self.scale_label.setText("No calibration yet -- nothing can be measured.")

        results = self.window.results
        self._fill(self.measurements, measurement_rows(results))

        head = results["head"]
        gear = recommend_gear(results["segments"], results["height_cm"],
                              results["hand"], head)
        rows = []
        for item, size in gear.items():
            # recommend_gear appends any head warning inline for the text
            # path; here it gets its own row instead of being repeated.
            rows.append((item.replace("_", " ").title(), size.split("  [!] ")[0]))
        if head and head.get("warning"):
            rows.append(("Note", head["warning"]))
        self._fill(self.recommendations, rows)


class MainWindow(QtWidgets.QMainWindow):
    """Sidebar of steps on the left, the active step on the right."""

    STATE_MARKS = {"done": "✓", "attention": "!", "skipped": "–", "empty": "·"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Measure AI -- Cricket Gear Sizing")
        self.resize(1180, 820)

        self.photos = {}
        self.skipped = set()
        self.pixels_per_cm = None
        self.scale_source = ""
        self.results = empty_results()

        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(250)
        self.sidebar.currentRowChanged.connect(self._change_page)

        self.pages = QtWidgets.QStackedWidget()
        self.step_pages = []
        for step in CAPTURE_STEPS:
            page = StepPage(step, self)
            self.step_pages.append(page)
            self.pages.addWidget(page)
            self.sidebar.addItem("")

        self.results_page = ResultsPage(self)
        self.pages.addWidget(self.results_page)
        self.sidebar.addItem("")

        splitter = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(splitter)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.pages, stretch=1)
        self.setCentralWidget(splitter)

        self.statusBar().showMessage(
            "Start with the A4 sheet -- its scale is what turns pixels into centimetres.")

        self._adopt_existing_photos()
        self.sidebar.setCurrentRow(0)
        self._refresh_sidebar()

    def _adopt_existing_photos(self):
        """Restores the previous run's photos, wherever they live."""
        self.photos = load_session_photos()
        for page in self.step_pages:
            page.refresh()
        if "calibration" in self.photos:
            self.step_pages[0].measure()

    # -- session state -----------------------------------------------------

    def set_scale(self, pixels_per_cm, source="", warning=None):
        changed = pixels_per_cm != self.pixels_per_cm
        self.pixels_per_cm = pixels_per_cm
        self.scale_source = source
        if warning:
            QtWidgets.QMessageBox.warning(self, "Check the calibration photo", warning)
        if pixels_per_cm:
            self.statusBar().showMessage(f"Scale: {pixels_per_cm:.2f} px/cm ({source})")
        self._refresh_sidebar()
        self.results_page.refresh()
        # Every measurement is expressed in this scale, so a new one
        # invalidates all of them.
        if changed and pixels_per_cm:
            self.measure_all(skip_calibration=True)

    def measure_all(self, skip_calibration=False):
        for page in self.step_pages:
            if skip_calibration and page.is_calibration:
                continue
            if page.key in self.photos and page.key not in self.results["manual"]:
                page.measure()

    def step_changed(self, key):
        self._refresh_sidebar()
        self.results_page.refresh()

    def _refresh_sidebar(self):
        for index, page in enumerate(self.step_pages):
            mark = self.STATE_MARKS[page.state()]
            self.sidebar.item(index).setText(f" {mark}  {index + 1}. {page.step['label']}")
        self.sidebar.item(len(self.step_pages)).setText("     Results")

    def _change_page(self, row):
        if row < 0:
            return
        for page in self.step_pages:
            page.closing()
        self.pages.setCurrentIndex(row)
        if row < len(self.step_pages):
            self.step_pages[row].refresh()
        else:
            self.results_page.refresh()

    def closeEvent(self, event):
        for page in self.step_pages:
            page.closing()
        super().closeEvent(event)


def run_gui():
    application = QtWidgets.QApplication(sys.argv)
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
