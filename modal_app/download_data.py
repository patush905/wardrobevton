"""
One-off: download the try-on datasets into the "wardrobe-data" Modal Volume.

    modal run modal_app/download_data.py::download            # VITON-HD (tops)
    modal run modal_app/download_data.py::download_omniview   # OmniView (tops, bottoms, full body)

Runs on a CPU container (no GPU billing). The archive goes to the container's
local disk first, then is extracted into the Volume, so the Volume ends up
with only the extracted files under /viton-hd/zalando-hd-resized:

    train/{image,cloth,cloth-mask,agnostic-mask,image-densepose,openpose_json,...}
    test/...
    train_pairs.txt, test_pairs.txt, test_unpairs.txt

Source: the official zalando-hd-resized.tar.gz, mirrored on Hugging Face by
the CatVTON author. The Google Drive link in the VITON-HD README hits Drive's
"too many accesses" block. VITON-HD is for research purposes only - see
https://github.com/shadow2496/VITON-HD
"""

import modal

app = modal.App("wardrobe-data")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")

data = modal.Volume.from_name("wardrobe-data", create_if_missing=True)

REPO = "zhengchong/VITON-HD"
ARCHIVE = "zalando-hd-resized.tar.gz"
DEST = "/data/viton-hd"


@app.function(image=image, volumes={"/data": data}, timeout=2 * 3600)
def download():
    import os
    import subprocess

    from huggingface_hub import hf_hub_download

    if os.path.isdir(DEST) and os.listdir(DEST):
        print(f"{DEST} already has data, skipping")
        return

    # Local container disk, not the Volume: we only want the extracted files kept.
    archive = hf_hub_download(REPO, ARCHIVE, repo_type="dataset", local_dir="/tmp/viton-hd")

    os.makedirs(DEST, exist_ok=True)
    subprocess.run(["tar", "-xzf", archive, "-C", DEST], check=True)

    # Make the files visible to other containers (e.g. the training job).
    data.commit()
    for root, dirs, files in os.walk(DEST):
        depth = root[len(DEST) :].count(os.sep)
        if depth <= 2:
            print(f"{root}: {len(dirs)} dirs, {len(files)} files")


# OmniView is gated: accept its terms on huggingface.co with your account, then
# store a read token as a Modal Secret named "hf-omniview" (key HF_TOKEN).
# Licence is CC BY-NC-SA 4.0 - non-commercial only.
OMNIVIEW_REPO = "zwpro/OmniView"
OMNIVIEW_DEST = "/data/omniview"


@app.function(
    image=image,
    volumes={"/data": data},
    secrets=[modal.Secret.from_name("hf-omniview")],
    timeout=4 * 3600,
)
def download_omniview():
    import os

    from huggingface_hub import snapshot_download

    # ~60k small files (images + anno.json per sample), so write straight into
    # the Volume rather than via an archive. snapshot_download skips files that
    # are already there, so a rerun after a timeout resumes instead of restarting.
    snapshot_download(
        OMNIVIEW_REPO, repo_type="dataset", local_dir=OMNIVIEW_DEST, max_workers=16
    )

    data.commit()
    for name in sorted(os.listdir(OMNIVIEW_DEST)):
        path = os.path.join(OMNIVIEW_DEST, name)
        if os.path.isdir(path):
            print(f"{name}: {sorted(os.listdir(path))}")
