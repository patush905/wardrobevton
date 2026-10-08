# catvton_modal.py
import modal

app = modal.App("catvton")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/Zheng-Chong/CatVTON /repo",
        "sed -i '/diffusers/d' /repo/requirements.txt",
        "pip install -r /repo/requirements.txt 'diffusers==0.35.1'",
    )
    .env({"HF_HOME": "/weights/hf"})  # HF cache lives in the volume
)

weights = modal.Volume.from_name("tryon-weights", create_if_missing=True)

BASE = "booksforcharlie/stable-diffusion-inpainting"
CKPT = "zhengchong/CatVTON"


@app.function(image=image, volumes={"/weights": weights}, timeout=3600)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(CKPT)
    snapshot_download(BASE)
    weights.commit()


@app.cls(gpu="A10G", image=image, volumes={"/weights": weights})
class CatVTON:
    @modal.enter()
    def load(self):
        import sys, torch

        sys.path.insert(0, "/repo")
        from huggingface_hub import snapshot_download
        from diffusers.image_processor import VaeImageProcessor
        from model.pipeline import CatVTONPipeline

        repo_path = snapshot_download(CKPT)  # already cached, so this is instant
        self.pipe = CatVTONPipeline(
            base_ckpt=BASE,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=torch.bfloat16,
            use_tf32=True,
            device="cuda",
        )
        self.mask_proc = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    @modal.method()
    def run(self, person: bytes, cloth: bytes, mask: bytes) -> bytes:
        import io, torch
        from PIL import Image
        from utils import resize_and_crop, resize_and_padding

        W, H = 768, 1024
        p = resize_and_crop(Image.open(io.BytesIO(person)).convert("RGB"), (W, H))
        c = resize_and_padding(Image.open(io.BytesIO(cloth)).convert("RGB"), (W, H))
        m = resize_and_crop(Image.open(io.BytesIO(mask)).convert("L"), (W, H))
        m = self.mask_proc.blur(m, blur_factor=9)

        out = self.pipe(
            image=p,
            condition_image=c,
            mask=m,
            num_inference_steps=50,
            guidance_scale=2.5,
            generator=torch.Generator("cuda").manual_seed(42),
        )[0]
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return buf.getvalue()


@app.local_entrypoint()
def main(person: str, cloth: str, mask: str):
    r = lambda f: open(f, "rb").read()
    out = CatVTON().run.remote(r(person), r(cloth), r(mask))
    open("result.png", "wb").write(out)
