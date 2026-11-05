"""
One-off: pull the Leffa checkpoints from Hugging Face into a Modal Volume.

    modal run modal_app/download_weights.py

Run once with `modal run`, not on every container start. Weights in the image
mean every rebuild re-downloads several GB and cold starts get worse; weights
in a Volume are mounted instantly.

franciszzj/Leffa has separate checkpoints for VITON-HD try-on, DressCode
try-on and DeepFashion pose transfer. We only take the two try-on ones plus
the preprocessing models (~16 GB); pose transfer alone is another 21 GB.
"""

import modal

app = modal.App("wardrobe-leffa-weights")

image = modal.Image.debian_slim(python_version="3.10").pip_install("huggingface_hub>=0.34,<1.0")

weights = modal.Volume.from_name("tryon-weights", create_if_missing=True)

# Must match LEFFA_DIR in leffa_app.py.
LEFFA_DIR = "/weights/leffa"

ALLOW = [
    "virtual_tryon.pth",  # VITON-HD model: tops only
    "virtual_tryon_dc.pth",  # DressCode model: upper_body, lower_body, dresses
    "stable-diffusion-inpainting/*",  # just configs; the weights are inside the .pth files
    "densepose/*",  # body surface map, conditions the generation
    "humanparsing/*",  # pixel labels, used to build the agnostic mask
    "openpose/*",  # keypoints, also used for the agnostic mask
]


@app.function(image=image, volumes={"/weights": weights}, timeout=2 * 3600)
def download():
    import os

    from huggingface_hub import snapshot_download

    # Skips files already present, so a rerun after a timeout resumes.
    snapshot_download("franciszzj/Leffa", local_dir=LEFFA_DIR, allow_patterns=ALLOW)
    weights.commit()

    for name in sorted(os.listdir(LEFFA_DIR)):
        path = os.path.join(LEFFA_DIR, name)
        size = os.path.getsize(path) if os.path.isfile(path) else None
        print(name, f"{size / 1e9:.1f} GB" if size else "/")
