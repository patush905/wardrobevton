"""Live pose landmark detection from webcam.

Opens the default camera via OpenCV, runs MediaPipe PoseLandmarker in VIDEO
mode on each frame, and shows an annotated preview window. With --segment it
also runs the multiclass selfie segmenter, smooths the masks over frames, and
cleans them before drawing.

Run: python -m server.mapping.livelandmarker [--camera 0] [--model models/pose_landmarker_full.task]
     [--segment] [--seg-model models/selfie_multiclass_256x256.tflite]
Press m to cycle the segmentation overlay, q (or ESC) to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from server.mapping import segmentation as seg

DEFAULT_MODEL = Path("models/pose_landmarker_full.task")

POSE_CONNECTIONS = frozenset(mp.solutions.pose.POSE_CONNECTIONS)

# Segmentation views, cycled with the m key:
#   classes - every class in its own color
#   person  - cleaned whole-body mask
#   clothes - cleaned clothes mask
#   off     - skip segmentation entirely (faster)
OVERLAY_MODES = ("classes", "person", "clothes", "off")


def _draw_overlay(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    thickness = max(2, min(w, h) // 300)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), thickness)

    r = max(3, min(w, h) // 200)
    for x, y in pts:
        cv2.circle(frame, (x, y), r, (0, 0, 255), -1)


def _draw_segmentation(frame, result, smoothers: dict, mode: str) -> None:
    """Draw the segmentation for `mode` onto `frame` in place."""
    if mode == "classes":
        # Raw category mask, no cleanup: 60% camera image, 40% class colors.
        # frame[:] = ... writes into the existing array instead of rebinding.
        frame[:] = cv2.addWeighted(frame, 0.6, seg.colorize(seg.category_mask(result)), 0.4, 0)
        return

    person, clothes = seg.person_and_clothes(seg.confidences(result))
    # Average with previous frames first (stops flicker), then clean up.
    soft = smoothers[mode](person if mode == "person" else clothes)
    # Person keeps only the largest blob; clothes keep any blob >= 5% of it.
    _, alpha = seg.clean_mask(soft, min_component_ratio=1.0 if mode == "person" else 0.05)

    # Dim everything outside the mask and tint the inside.
    tint = np.array((255, 160, 0) if mode == "clothes" else (0, 200, 0), dtype=np.float32)
    # alpha is HxW; adding an axis makes it HxWx1 so it multiplies all 3
    # color channels. Outside the mask (a=0) the frame drops to 35%
    # brightness; inside (a=1) it stays full brightness plus some tint.
    a = alpha[..., None]
    out = frame * (0.35 + 0.65 * a) + tint * (0.25 * a)
    frame[:] = np.clip(out, 0, 255).astype(np.uint8)


def run(
    camera_index: int, model_path: Path, mirror: bool, seg_model: Path | None = None
) -> int:
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )

    # Only build the segmenter when --segment was passed. VIDEO mode matches
    # the pose landmarker and needs increasing timestamps per frame.
    segmenter = (
        seg.create_segmenter(seg_model, mp_vision.RunningMode.VIDEO) if seg_model else None
    )
    # One smoother per mask, so switching views doesn't blend person and
    # clothes masks together.
    smoothers = {"person": seg.MaskSmoother(), "clothes": seg.MaskSmoother()}
    mode_idx = 0  # index into OVERLAY_MODES; starts on "classes"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"could not open camera {camera_index}", file=sys.stderr)
        return 2

    start = time.monotonic()
    last_fps_log = start
    frames = 0

    try:
        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("failed to read frame", file=sys.stderr)
                    break

                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts_ms = int((time.monotonic() - start) * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                # Segment before drawing the skeleton so the skeleton stays
                # on top. Reuses the same mp_image and timestamp as pose.
                if segmenter and OVERLAY_MODES[mode_idx] != "off":
                    seg_result = segmenter.segment_for_video(mp_image, ts_ms)
                    _draw_segmentation(frame, seg_result, smoothers, OVERLAY_MODES[mode_idx])

                if result.pose_landmarks:
                    _draw_overlay(frame, result.pose_landmarks[0])

                frames += 1
                now = time.monotonic()
                if now - last_fps_log >= 1.0:
                    fps = frames / (now - last_fps_log)
                    # Show the current overlay mode in the window title.
                    overlay = f"  [{OVERLAY_MODES[mode_idx]}, m to cycle]" if segmenter else ""
                    cv2.setWindowTitle("pose", f"pose  —  {fps:.1f} fps{overlay}  (q to quit)")
                    frames = 0
                    last_fps_log = now

                cv2.imshow("pose", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                # m cycles to the next overlay, wrapping back to the start.
                if key == ord("m"):
                    mode_idx = (mode_idx + 1) % len(OVERLAY_MODES)
    finally:
        # The segmenter isn't in a `with` block, so release it explicitly.
        if segmenter:
            segmenter.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live pose landmarks from a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to .task file")
    parser.add_argument(
        "--no-mirror",
        dest="mirror",
        action="store_false",
        help="disable the default horizontal flip",
    )
    parser.add_argument(
        "--segment", action="store_true", help="also show person/clothes segmentation"
    )
    parser.add_argument(
        "--seg-model", type=Path, default=seg.DEFAULT_MODEL, help="path to segmenter .tflite"
    )
    args = parser.parse_args(argv)
    # Passing None as seg_model turns segmentation off.
    return run(args.camera, args.model, args.mirror, args.seg_model if args.segment else None)


if __name__ == "__main__":
    raise SystemExit(main())
