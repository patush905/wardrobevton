"""Person/clothing segmentation via MediaPipe's multiclass selfie segmenter.

The model labels every pixel as one of CLASSES. We turn its soft per-class
confidence maps into clean masks with clean_mask():
binarize -> open/close -> drop stray blobs -> fill small holes -> feather.

Run: python -m server.mapping.segmentation <image_path> [--out out_dir]
Writes <stem>_steps.png (each cleanup step side by side), plus the cleaned
person/clothes masks and a transparent clothes cutout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Downloaded from Google's MediaPipe model storage (float32 build).
DEFAULT_MODEL = Path("models/selfie_multiclass_256x256.tflite")

# Output channel order of selfie_multiclass_256x256. The model returns one
# confidence map per entry here, in this exact order, so index 4 is clothes.
CLASSES = ("background", "hair", "body_skin", "face_skin", "clothes", "others")

# BGR colors for visualizing the category mask (OpenCV uses BGR, not RGB).
# Row i is the color for CLASSES[i], so we can index this array with the
# category mask directly to paint every pixel at once.
CLASS_COLORS = np.array(
    [
        (0, 0, 0),  # background
        (0, 200, 255),  # hair
        (140, 180, 255),  # body skin
        (180, 120, 255),  # face skin
        (255, 160, 0),  # clothes
        (0, 255, 120),  # others (accessories)
    ],
    dtype=np.uint8,
)


def create_segmenter(
    model_path: Path = DEFAULT_MODEL,
    running_mode: mp_vision.RunningMode = mp_vision.RunningMode.IMAGE,
) -> mp_vision.ImageSegmenter:
    """Build a segmenter. Use RunningMode.VIDEO for webcam frames.

    The returned object holds native resources, so use it in a `with` block
    or call .close() when done.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    options = mp_vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        # category_mask: one class index per pixel (the "winning" class).
        output_category_mask=True,
        # confidence_masks: one 0..1 map per class. These are what we clean,
        # because thresholding a soft map gives smoother edges than the
        # hard category mask.
        output_confidence_masks=True,
    )
    return mp_vision.ImageSegmenter.create_from_options(options)


def confidences(result) -> dict[str, np.ndarray]:
    """Per-class float32 confidence maps (HxW, 0..1), keyed by class name."""
    # .copy() because numpy_view() points into MediaPipe's memory, which is
    # reused on the next frame.
    return {name: m.numpy_view().copy() for name, m in zip(CLASSES, result.confidence_masks)}


def category_mask(result) -> np.ndarray:
    """HxW uint8 array of class indices into CLASSES."""
    return result.category_mask.numpy_view().copy()


def person_and_clothes(conf: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Soft person mask (anything not background) and soft clothes mask."""
    # Person = "not background", which covers hair + skin + clothes + others
    # without having to add all of those maps together.
    return 1.0 - conf["background"], conf["clothes"]


def colorize(categories: np.ndarray) -> np.ndarray:
    """Turn a category mask into a BGR image using CLASS_COLORS."""
    # Fancy indexing: each pixel's class index picks a row of CLASS_COLORS.
    # clip() guards against any out-of-range value crashing the lookup.
    return CLASS_COLORS[np.clip(categories, 0, len(CLASSES) - 1)]


def _odd(n: float) -> int:
    """Round to a positive odd integer. OpenCV kernels need odd sizes so they
    have a center pixel."""
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def clean_mask(
    mask: np.ndarray,
    thresh: float = 0.5,
    kernel_size: int | None = None,
    min_component_ratio: float = 1.0,
    min_hole_area: int | None = None,
    feather: int | None = None,
    return_steps: bool = False,
):
    """Clean a soft (0..1) or binary (0/255 uint8) mask.

    kernel_size: morphology kernel; defaults to ~0.75% of image width.
    min_component_ratio: keep blobs at least this fraction of the largest blob's
        area. 1.0 keeps only the largest (one person); lower it for masks that
        are legitimately split, like a shirt and shoes.
    min_hole_area: interior holes smaller than this get filled; defaults to
        0.1% of the image area, so real gaps (arm vs. torso) survive.
    feather: Gaussian blur size for the soft edge; defaults to ~1% of width.

    Returns (hard, alpha): hard is uint8 0/255 for geometry (contours, boxes,
    warping), alpha is float32 0..1 for compositing. With return_steps=True,
    also returns a list of (name, uint8 image) for each stage.
    """
    h, w = mask.shape[:2]
    # Defaults scale with resolution, so the same call works on a 400px
    # thumbnail and a 4000px photo.
    kernel_size = _odd(kernel_size or w * 0.0075)
    feather = _odd(feather or w * 0.01)
    if min_hole_area is None:
        min_hole_area = int(h * w * 0.001)

    # Snapshots of the mask after each stage, for the side-by-side debug view.
    steps: list[tuple[str, np.ndarray]] = []

    # 1. Binarize: a soft float map becomes a 0/255 mask. Pixels above
    #    `thresh` confidence count as "in". Already-binary uint8 masks skip this.
    if mask.dtype != np.uint8:
        steps.append(("soft", (np.clip(mask, 0, 1) * 255).astype(np.uint8)))
        mask = (mask > thresh).astype(np.uint8) * 255
    steps.append(("binarized", mask))

    # 2. Morphology. An elliptical kernel avoids boxy artifacts on curved edges.
    #    Open (erode then dilate) deletes specks smaller than the kernel.
    #    Close (dilate then erode) seals thin cracks; two passes for stubborn ones.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    steps.append(("open/close", mask))

    # 3. Drop stray blobs. connectedComponentsWithStats gives every separate
    #    white region its own label (0 is the black background) plus its area.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip label 0 (background)
        # keep[label] says whether that blob survives; background stays False.
        keep = np.zeros(n, dtype=bool)
        keep[1:] = areas >= areas.max() * min_component_ratio
        # keep[labels] maps the per-label decision onto every pixel at once.
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    steps.append(("blobs removed", mask))

    # 4. Fill small interior holes. Holes are blobs of the *inverted* mask.
    #    A region touching the image border is outside space, not a hole.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(mask), connectivity=8
    )
    if n > 1:
        # Bounding box (x, y, width, height) and area for each region.
        x, y, bw, bh, area = (stats[:, i] for i in range(5))
        interior = (x > 0) & (y > 0) & (x + bw < w) & (y + bh < h)
        # Only fill small holes; big ones (e.g. the gap between arm and
        # torso) are real and should stay empty.
        fill = interior & (area < min_hole_area)
        fill[0] = False  # label 0 is the mask itself, not a hole
        mask = np.where(fill[labels], 255, mask).astype(np.uint8)
    steps.append(("holes filled", mask))

    # 5. Smooth jagged edges: blur, then re-threshold at the midpoint. This
    #    rounds off stair-step pixels while keeping the mask strictly 0/255.
    hard = (cv2.GaussianBlur(mask, (feather, feather), 0) > 127).astype(np.uint8) * 255
    steps.append(("smoothed", hard))

    # 6. Soft alpha: blur the clean hard mask so the edge fades over a few
    #    pixels. Compositing with this looks far more natural than a hard cut.
    alpha = cv2.GaussianBlur(hard, (feather, feather), 0).astype(np.float32) / 255
    steps.append(("alpha", (alpha * 255).astype(np.uint8)))

    if return_steps:
        return hard, alpha, steps
    return hard, alpha


class MaskSmoother:
    """Exponential moving average over frames, to stop mask flicker on video.

    Each output is `weight * previous + (1 - weight) * new`. Higher weight means
    steadier masks but more lag when you move. Use one smoother per mask type.
    """

    def __init__(self, weight: float = 0.6):
        self.weight = weight  # how much of the previous frame to keep
        self._state: np.ndarray | None = None  # running average so far

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        # First frame (or the camera resolution changed): nothing to blend
        # with yet, so start the average from this frame.
        if self._state is None or self._state.shape != mask.shape:
            self._state = mask.astype(np.float32)
        else:
            self._state = self.weight * self._state + (1 - self.weight) * mask
        return self._state


def segment_image(image_bgr: np.ndarray, segmenter: mp_vision.ImageSegmenter):
    """Run an IMAGE-mode segmenter on an OpenCV (BGR) image."""
    # MediaPipe expects RGB; OpenCV loads BGR. Building the mp.Image ourselves
    # (instead of mp.Image.create_from_file) also avoids PNGs loading as RGBA.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Return a BGR copy of img with `text` written in the top-left corner."""
    # Masks are single-channel; convert so every tile in the grid is BGR.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()  # don't draw on the caller's image
    # Scale the font with image width so labels stay readable.
    scale = max(0.5, img.shape[1] / 600)
    cv2.putText(img, text, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    return img


def steps_grid(image_bgr: np.ndarray, categories: np.ndarray, steps, cols: int = 4) -> np.ndarray:
    """Original, class overlay, and every clean_mask stage, tiled into one image."""
    # 50/50 blend of the photo and the class colors.
    overlay = cv2.addWeighted(image_bgr, 0.5, colorize(categories), 0.5, 0)
    tiles = [_label(image_bgr, "original"), _label(overlay, "classes")]
    tiles += [_label(img, name) for name, img in steps]

    # Resize every tile to the same width (keeping aspect ratio) so they stack.
    tile_w = 400
    h, w = image_bgr.shape[:2]
    tile_h = int(h * tile_w / w)
    tiles = [cv2.resize(t, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for t in tiles]
    # Pad the last row with black tiles so every row has `cols` tiles.
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    # Join tiles side by side into rows, then stack the rows vertically.
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def cutout(image_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """BGRA image with alpha applied (for saving as PNG)."""
    # Add a 4th (alpha) channel and fill it from the mask: 255 = opaque,
    # 0 = transparent. Only PNG keeps this channel; JPEG would drop it.
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = (alpha * 255).astype(np.uint8)
    return bgra


def write_outputs(image_path: Path, out_dir: Path, model_path: Path = DEFAULT_MODEL) -> list[Path]:
    """Segment one image and write the debug grid, masks, and cutout.

    Returns the paths written.
    """
    image = cv2.imread(str(image_path))
    if image is None:  # cv2.imread returns None instead of raising
        raise FileNotFoundError(f"could not read image: {image_path}")

    with create_segmenter(model_path) as segmenter:
        result = segment_image(image, segmenter)

    person_soft, clothes_soft = person_and_clothes(confidences(result))
    # Person: one body, so keep only the largest blob (the default).
    person_hard, _ = clean_mask(person_soft)
    # Clothes can be split into pieces (e.g. top and pants separated by skin),
    # so keep any blob at least 5% the size of the largest one. We also ask
    # for the intermediate steps to build the debug grid.
    clothes_hard, clothes_alpha, steps = clean_mask(
        clothes_soft, min_component_ratio=0.05, return_steps=True
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    # Output path -> image to save there.
    outputs = {
        out_dir / f"{stem}_steps.png": steps_grid(image, category_mask(result), steps),
        out_dir / f"{stem}_person_mask.png": person_hard,
        out_dir / f"{stem}_clothes_mask.png": clothes_hard,
        out_dir / f"{stem}_clothes_cutout.png": cutout(image, clothes_alpha),
    }
    for path, img in outputs.items():
        cv2.imwrite(str(path), img)
    return list(outputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segment person/clothes and clean the masks.")
    parser.add_argument("image", type=Path, help="path to input image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="path to .tflite")
    parser.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = parser.parse_args(argv)

    for path in write_outputs(args.image, args.out, args.model):
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
