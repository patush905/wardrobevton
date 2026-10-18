"""Split the clothes mask into a top and a bottom using the pose hip line.

MediaPipe's segmenter has a single "clothes" class, so a shirt and jeans come
out as one mask. The pose landmarker gives us the two hips (23 = left,
24 = right); the line through them is where tops end and bottoms begin.

These are the masks CatVTON repaints: the upper mask for a top, the lower
mask for pants/skirts, the full clothes mask for a dress or outfit.

Run: python -m server.mapping.garment_mask <image_path> [--out out_dir]
Writes <stem>_upper_mask.png, <stem>_lower_mask.png and <stem>_split.png
(the photo with each region tinted and the hip line drawn).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the two pieces we already have: pose detection from imagelandmarker
# and mask cleanup from segmentation. This module only adds the split.
from server.mapping.imagelandmarker import DEFAULT_MODEL as POSE_MODEL
from server.mapping.imagelandmarker import detect
from server.mapping.segmentation import (
    clean_mask,
    confidences,
    create_segmenter,
    person_and_clothes,
    segment_image,
)

# MediaPipe pose landmark indices (there are 33 in total). "Left" and "right"
# are the person's own left/right, so on a front-facing photo LEFT_HIP is on
# the right side of the image.
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

# Every landmark comes with a 0..1 visibility score. Below this the landmarker
# is guessing (hips hidden behind something or out of frame), and a split
# along a guessed line is worse than no split at all.
MIN_VISIBILITY = 0.5

# BGR tints for the debug overlay (OpenCV uses BGR, not RGB).
UPPER_COLOR = (255, 160, 0)  # blue
LOWER_COLOR = (0, 160, 255)  # orange


class PoseError(ValueError):
    """The photo can't be split: no person found, or the hips aren't visible."""


def hip_line(landmarks: list[dict], w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (x, y) of the left and right hip."""
    hips = [landmarks[LEFT_HIP], landmarks[RIGHT_HIP]]
    # Both hips have to be trustworthy; one good hip doesn't give us a line.
    if min(lm["visibility"] for lm in hips) < MIN_VISIBILITY:
        raise PoseError("hips not clearly visible")
    # Landmarks are normalized to 0..1 of the image size, so scale them to
    # pixels to line up with the mask.
    return tuple(np.array([lm["x"] * w, lm["y"] * h]) for lm in hips)


def torso_length(landmarks: list[dict], w: int, h: int) -> float:
    """Shoulder-midpoint to hip-midpoint distance in pixels. Used to scale the
    overlap margin, so it means the same thing for near and far subjects."""
    # Same normalized -> pixel conversion as hip_line, for any landmark index.
    pt = lambda i: np.array([landmarks[i]["x"] * w, landmarks[i]["y"] * h])  # noqa: E731
    shoulders = (pt(LEFT_SHOULDER) + pt(RIGHT_SHOULDER)) / 2
    hips = (pt(LEFT_HIP) + pt(RIGHT_HIP)) / 2
    # Straight-line (Euclidean) distance between the two midpoints.
    return float(np.linalg.norm(hips - shoulders))


def above_line(a: np.ndarray, b: np.ndarray, w: int, h: int, offset: float = 0.0) -> np.ndarray:
    """HxW bool mask of pixels above the line through a and b, shifted down by
    `offset` pixels. Following the actual line (not a flat y) keeps the split
    right when the hips are tilted."""
    xs = np.arange(w)
    # Slope of the hip line. Hips are never vertically stacked in a standing
    # photo, but guard the division anyway.
    dx = b[0] - a[0] if abs(b[0] - a[0]) > 1e-6 else 1e-6
    # The line's y at every column x (point-slope form), then shifted down by
    # `offset`. Image y grows downward, so a positive offset moves it lower.
    line_y = a[1] + (xs - a[0]) * (b[1] - a[1]) / dx + offset
    # Broadcasting: a column of row numbers (h, 1) against a row of line
    # heights (1, w) compares every pixel to its column's line in one step.
    return np.arange(h)[:, None] < line_y[None, :]


def split_clothes(
    clothes: np.ndarray, landmarks: list[dict], overlap: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    """Split a 0/255 clothes mask into (upper, lower) at the hip line.

    overlap: fraction of torso length each mask extends past the hip line.
    Shirts often hang below the hips and waistbands sit above them, and CatVTON
    does better with a slightly generous mask than one that clips the garment,
    so the two masks deliberately overlap around the waist.
    """
    h, w = clothes.shape[:2]
    left, right = hip_line(landmarks, w, h)
    margin = overlap * torso_length(landmarks, w, h)

    # Upper: clothes above a line pushed `margin` BELOW the hips.
    upper = np.where(above_line(left, right, w, h, +margin), clothes, 0).astype(np.uint8)
    # Lower: clothes below a line pulled `margin` ABOVE the hips. `~` flips
    # the bool mask from "above" to "below".
    lower = np.where(~above_line(left, right, w, h, -margin), clothes, 0).astype(np.uint8)
    return upper, lower


def split_overlay(image_bgr: np.ndarray, upper: np.ndarray, lower: np.ndarray, landmarks) -> np.ndarray:
    """The photo with the upper region tinted blue, lower orange (overlap shows
    both), and the hip line drawn in red."""
    out = image_bgr.copy()  # don't draw on the caller's image
    for mask, color in ((upper, UPPER_COLOR), (lower, LOWER_COLOR)):
        # A solid image of the tint color, blended 60/40 with the photo, but
        # copied back only where this mask is set. Doing upper then lower means
        # the overlap band gets both tints.
        tint = np.zeros_like(out)
        tint[:] = color
        sel = mask > 0
        out[sel] = cv2.addWeighted(out, 0.6, tint, 0.4, 0)[sel]
    h, w = out.shape[:2]
    a, b = hip_line(landmarks, w, h)
    # cv2.line wants integer pixel tuples; thickness scales with image width.
    cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), (0, 0, 255), max(2, w // 200))
    return out


def garment_masks(image_path: Path, pose_model: Path = POSE_MODEL):
    """Run pose + segmentation on one photo.

    Returns (image_bgr, upper, lower, clothes, landmarks); masks are 0/255.
    Raises PoseError if there's no usable person - treat that as a photo
    quality rejection before spending GPU time.
    """
    image = cv2.imread(str(image_path))
    if image is None:  # cv2.imread returns None instead of raising
        raise FileNotFoundError(f"could not read image: {image_path}")

    # Pose first: it's the cheap check. No person means no point segmenting.
    pose = detect(image_path, pose_model)
    if not pose["detected"]:
        raise PoseError("no person detected")

    with create_segmenter() as segmenter:
        result = segment_image(image, segmenter)
    # We only need the clothes map here; the person map is ignored.
    _, clothes_soft = person_and_clothes(confidences(result))
    # Same settings as segmentation.write_outputs: keep top + pants as two blobs.
    clothes, _ = clean_mask(clothes_soft, min_component_ratio=0.05)

    upper, lower = split_clothes(clothes, pose["landmarks"])
    return image, upper, lower, clothes, pose["landmarks"]


def write_outputs(image_path: Path, out_dir: Path) -> list[Path]:
    """Write the upper/lower masks and the debug overlay. Returns the paths."""
    image, upper, lower, _, landmarks = garment_masks(image_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    # Output path -> image to save there.
    outputs = {
        out_dir / f"{stem}_upper_mask.png": upper,
        out_dir / f"{stem}_lower_mask.png": lower,
        out_dir / f"{stem}_split.png": split_overlay(image, upper, lower, landmarks),
    }
    for path, img in outputs.items():
        cv2.imwrite(str(path), img)
    return list(outputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split the clothes mask into top and bottom.")
    parser.add_argument("image", type=Path, help="path to input image")
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = parser.parse_args(argv)

    try:
        paths = write_outputs(args.image, args.out)
    except PoseError as e:
        # An expected outcome for a bad photo, so a clean message, not a traceback.
        print(f"can't split {args.image}: {e}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
