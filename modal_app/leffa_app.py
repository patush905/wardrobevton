"""
The GPU service: Leffa virtual try-on on Modal.

    modal run modal_app/download_weights.py        # once: weights into the Volume
    modal run modal_app/leffa_app.py --person person.jpg --cloth top.jpg
    modal run modal_app/leffa_app.py --person person.jpg --cloth jeans.jpg --garment lower_body

Structure:
  - modal.Image with torch, diffusers and the Leffa repo (which vendors its
    own detectron2 + densepose, so no detectron2 build)
  - the "tryon-weights" Volume holding the checkpoints (see download_weights.py)
  - @app.cls so the models load once per container, in @modal.enter()

Leffa builds its own "agnostic" mask (the region to regenerate) from human
parsing + OpenPose, generous and with face/hands kept, which is what it was
trained on. Pass --mask and --pose to use ours instead: the garment mask and
MediaPipe landmarks from server/mapping/garment_mask.py. The pose takes
OpenPose's job (arm strokes so sleeves can change, hands kept) and neither
SCHP nor OpenPose runs:

    python -m server.mapping.garment_mask person.jpg --out out/
    modal run modal_app/leffa_app.py --person person.jpg --cloth top.jpg \
        --mask out/person_upper_mask.png --pose out/person_pose.json

Two checkpoints:
  viton_hd   - VITON-HD, tops only, frontal women; the stronger model for tops
  dress_code - DressCode, upper_body / lower_body / dresses; use this for pants

Still to come (see server/vton/modal_leffa.py): a @modal.fastapi_endpoint that
takes signed image URLs, writes the result to Supabase Storage and calls back
to /internal/jobs/{id}/complete.

Cold starts are the thing to design around: expect 30s+ on the first request
after idle. scaledown_window keeps a container alive between requests;
min_containers=1 removes cold starts entirely at the cost of paying while idle.

Leffa's code is MIT licensed, but the checkpoints are trained on VITON-HD and
DressCode, which are research-only. Check those terms before commercial use.
"""

import modal

app = modal.App("wardrobe-leffa")

# Pinned so a rebuild gets the code this was tested against.
LEFFA_COMMIT = "05a259104b6927607776c7edb3e86b75406f20aa"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        # Leffa's requirements.txt is unpinned; these are versions that work together
        # (same torch/diffusers/accelerate stack as catvton_modal.py).
        "torch==2.4.0",
        "torchvision==0.19.0",
        "diffusers==0.35.1",
        "transformers==4.46.3",
        "accelerate==1.10.1",
        "huggingface_hub>=0.34,<1.0",
        "safetensors",
        "peft",
        "numpy==1.26.4",
        "opencv-python-headless",
        "pillow",
        "onnxruntime",  # human parsing runs as ONNX, on CPU
        "einops",
        "fvcore",
        "iopath",
        "pycocotools",
        "omegaconf",
        "cloudpickle",
        "scipy",
        "scikit-image",
        "matplotlib",
        "timm",
        "pyyaml",
        "regex",
        "tqdm",
        "psutil",
        "av",  # imported by the vendored densepose code
    )
    .run_commands(
        "git clone https://github.com/franciszzj/Leffa /leffa",
        f"cd /leffa && git checkout {LEFFA_COMMIT}",
    )
)

weights = modal.Volume.from_name("tryon-weights", create_if_missing=True)

# Must match LEFFA_DIR in download_weights.py.
LEFFA_DIR = "/weights/leffa"

W, H = 768, 1024  # Leffa's working resolution

# MediaPipe pose landmark indices, (shoulder, elbow, wrist) per arm, and the
# points outlining each hand: wrist, pinky, index, thumb.
ARMS = ((11, 13, 15), (12, 14, 16))
HANDS = ((15, 17, 19, 21), (16, 18, 20, 22))


def pose_mask(mask, landmarks: list[dict], size: tuple[int, int], garment: str, model: str):
    """Our garment mask plus arm strokes from MediaPipe pose, hands cut out.

    mask: the garment mask already fitted to W x H. landmarks: MediaPipe's 33
    points, normalized to the original photo, whose (width, height) is `size`.
    Does what get_agnostic_mask_hd does with OpenPose keypoints: an arm stroke
    shoulder -> elbow -> wrist so long sleeves can become short and vice versa,
    with hands kept so they come back unchanged.
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Same transform as resize_and_center: scale to fit, then center.
    w0, h0 = size
    s = min(W / w0, H / h0)
    left, top = (W - int(w0 * s)) // 2, (H - int(h0 * s)) // 2

    def pt(i):
        lm = landmarks[i]
        return int(left + lm["x"] * w0 * s), int(top + lm["y"] * h0 * s)

    def visible(*idx):
        return all(landmarks[i].get("visibility", 1.0) >= 0.5 for i in idx)

    m = (np.array(mask.convert("L")) > 127).astype(np.uint8) * 255
    # A little generous, like Leffa's dilated parse mask: a hem or collar that
    # segmentation clipped still gets repainted.
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=4)

    if garment in ("upper_body", "dresses"):
        # Leffa's widths: 60 (VITON-HD) / 45 (DressCode) at 512 px tall.
        width = int((60 if model == "viton_hd" else 45) / 512 * H)
        for arm in ARMS:
            if visible(*arm):
                cv2.polylines(m, [np.array([pt(i) for i in arm])], False, 255, width)
                cv2.circle(m, pt(arm[0]), width // 2, 255, -1)
        for hand in HANDS:
            if visible(*hand):
                hull = cv2.convexHull(np.array([pt(i) for i in hand]))
                keep = cv2.dilate(
                    cv2.fillConvexPoly(np.zeros_like(m), hull, 255), np.ones((5, 5), np.uint8), iterations=3
                )
                m[keep > 0] = 0

    return Image.fromarray(m)


@app.cls(gpu="L40S", image=image, volumes={"/weights": weights}, scaledown_window=300, timeout=600)
class Leffa:
    @modal.enter()
    def load(self):
        import sys

        # The repo root holds Leffa's packages plus its vendored detectron2/densepose.
        sys.path.insert(0, "/leffa")
        from leffa.inference import LeffaInference
        from leffa.model import LeffaModel
        from leffa_utils.densepose_predictor import DensePosePredictor
        from preprocess.humanparsing.run_parsing import Parsing
        from preprocess.openpose.run_openpose import OpenPose

        ck = LEFFA_DIR
        self.densepose = DensePosePredictor(
            config_path=f"{ck}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path=f"{ck}/densepose/model_final_162be9.pkl",
        )
        self.parsing = Parsing(
            atr_path=f"{ck}/humanparsing/parsing_atr.onnx",
            lip_path=f"{ck}/humanparsing/parsing_lip.onnx",
        )
        self.openpose = OpenPose(body_model_path=f"{ck}/openpose/body_pose_model.pth")

        # Both try-on models stay loaded (~7 GB each in fp16), so switching
        # between tops and pants doesn't mean a reload.
        self.inference = {
            name: LeffaInference(
                model=LeffaModel(
                    pretrained_model_name_or_path=f"{ck}/stable-diffusion-inpainting",
                    pretrained_model=f"{ck}/{ckpt}",
                    dtype="float16",
                )
            )
            for name, ckpt in (("viton_hd", "virtual_tryon.pth"), ("dress_code", "virtual_tryon_dc.pth"))
        }

    @modal.method()
    def run(
        self,
        person: bytes,
        cloth: bytes,
        garment: str = "upper_body",
        model: str = "",
        mask: bytes | None = None,
        pose: dict | None = None,
        steps: int = 50,
        scale: float = 2.5,
        seed: int = 42,
    ) -> dict:
        """Returns {"image": png bytes, "mask": png bytes of the mask used}.

        pose: MediaPipe output for the person photo, {"landmarks": [...33 points]},
        as written by garment_mask.py. Needs `mask` too.
        """
        import io

        import numpy as np
        from leffa.transform import LeffaTransform
        from leffa_utils.utils import get_agnostic_mask_dc, get_agnostic_mask_hd, resize_and_center
        from PIL import Image, ImageOps

        if garment not in ("upper_body", "lower_body", "dresses"):
            raise ValueError(f"unknown garment type: {garment}")
        # VITON-HD only knows tops, so anything else goes to the DressCode model.
        model = model or ("viton_hd" if garment == "upper_body" else "dress_code")
        if model == "viton_hd" and garment != "upper_body":
            raise ValueError("viton_hd only supports upper_body; use dress_code")
        if pose is not None and mask is None:
            raise ValueError("pose needs a mask: pass our garment mask with it")

        # Fit into 768x1024 without cropping: scale down, pad with white.
        person_img = Image.open(io.BytesIO(person))
        src = resize_and_center(person_img, W, H)
        ref = resize_and_center(Image.open(io.BytesIO(cloth)), W, H)
        src_array = np.array(src)

        if mask is not None:
            # Same fit as the person photo so it lines up. resize_and_center pads
            # with white (= "repaint"), so invert around it to pad with black.
            m = ImageOps.invert(Image.open(io.BytesIO(mask)).convert("L"))
            m = ImageOps.invert(resize_and_center(m, W, H).convert("L"))
            if pose is not None:
                m = pose_mask(m, pose["landmarks"], person_img.size, garment, model)
        else:
            # Leffa's own agnostic mask, computed at half resolution like upstream.
            small = src.resize((W // 2, H // 2))
            parse, _ = self.parsing(small)
            keypoints = self.openpose(small)
            agnostic = get_agnostic_mask_hd if model == "viton_hd" else get_agnostic_mask_dc
            m = agnostic(parse, keypoints, garment).resize((W, H))

        # DensePose conditioning; the two checkpoints were trained on different encodings.
        if model == "viton_hd":
            dp = self.densepose.predict_seg(src_array)[:, :, ::-1]
        else:
            iuv = self.densepose.predict_iuv(src_array)
            dp = np.concatenate([iuv[:, :, 0:1]] * 3, axis=-1)

        data = LeffaTransform()(
            {"src_image": [src], "ref_image": [ref], "mask": [m], "densepose": [Image.fromarray(dp)]}
        )
        out = self.inference[model](
            data, num_inference_steps=steps, guidance_scale=scale, seed=seed
        )["generated_image"][0]

        def png(img):
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue()

        return {"image": png(out), "mask": png(m)}


@app.local_entrypoint()
def main(
    person: str,
    cloth: str,
    garment: str = "upper_body",
    model: str = "",
    mask: str = "",
    pose: str = "",
    steps: int = 50,
    out: str = "result_leffa.png",
):
    import json

    r = lambda f: open(f, "rb").read()  # noqa: E731
    res = Leffa().run.remote(
        r(person),
        r(cloth),
        garment=garment,
        model=model,
        mask=r(mask) if mask else None,
        pose=json.load(open(pose)) if pose else None,
        steps=steps,
    )
    open(out, "wb").write(res["image"])
    open(out.replace(".png", "_mask.png"), "wb").write(res["mask"])
    print(f"saved {out} (+ _mask.png)")
