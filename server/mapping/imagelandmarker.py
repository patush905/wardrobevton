"""Pose landmark extraction via MediaPipe PoseLandmarker.

Run: python -m server.mapping.landmarks <image_path> [--model models/pose_landmarker_full.task]
Prints JSON: {"landmarks": [{"x","y","z","visibility","presence"}, ...]}.
Pass --draw out.png to also render landmarks + skeleton onto a copy of the image.
Pass --segment out_dir to also write cleaned person/clothes masks (see segmentation.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL = Path("models/pose_landmarker_full.task")

# Pairs of landmark indices that form the skeleton edges MediaPipe draws.
POSE_CONNECTIONS = frozenset(mp.solutions.pose.POSE_CONNECTIONS)


def detect(image_path: Path, model_path: Path) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(str(image_path))
        result = landmarker.detect(image)

    if not result.pose_landmarks:
        return {"landmarks": [], "detected": False}

    pose = result.pose_landmarks[0]
    return {
        "detected": True,
        "landmarks": [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
            for lm in pose
        ],
    }


def draw(image_path: Path, landmarks: list[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    canvas = ImageDraw.Draw(img)

    pts = [(lm["x"] * w, lm["y"] * h) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            canvas.line([pts[a], pts[b]], fill=(0, 255, 0), width=max(2, min(w, h) // 300))

    r = max(3, min(w, h) // 200)
    for x, y in pts:
        canvas.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))

    img.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract pose landmarks from an image.")
    parser.add_argument("image", type=Path, help="path to input image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to .task file")
    parser.add_argument("--draw", type=Path, help="write an annotated image to this path")
    parser.add_argument(
        "--segment", type=Path, metavar="DIR", help="write segmentation masks into this directory"
    )
    args = parser.parse_args(argv)

    out = detect(args.image, args.model)
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.draw and out["detected"]:
        draw(args.image, out["landmarks"], args.draw)
        print(f"wrote annotated image: {args.draw}", file=sys.stderr)

    # Segmentation is optional. Import it here so plain pose runs don't load
    # the segmentation module.
    if args.segment:
        from server.mapping.segmentation import write_outputs

        for path in write_outputs(args.image, args.segment):
            print(f"wrote {path}", file=sys.stderr)

    return 0 if out["detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
