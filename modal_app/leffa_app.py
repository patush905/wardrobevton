"""
The GPU service. Deployed separately:  modal deploy modal_app/leffa_app.py

Structure:
  - modal.App("wardrobe-leffa")
  - modal.Image with CUDA, torch, diffusers, detectron2, and the Leffa repo
  - modal.Volume holding the weights (see download_weights.py)
  - @app.cls(gpu="L40S", volumes=..., scaledown_window=...) so the model loads
    once per container instead of once per request
  - @modal.enter() loads the model into VRAM; containers are not considered
    warm until it returns
  - @modal.fastapi_endpoint() exposing POST /tryon

The endpoint takes two signed image URLs, downloads them, runs preprocessing
then Leffa, writes the result to Supabase Storage, and either returns the
image or calls back to your API.

Cold starts are the thing to design around: expect 30s+ on the first request
after idle. scaledown_window keeps a container alive between requests;
min_containers=1 removes cold starts entirely at the cost of paying while
idle. For a demo, warm it up a few minutes beforehand.

Leffa is MIT licensed, so unlike CatVTON and IDM-VTON there is no
non-commercial restriction on the code or weights. Check the training dataset
terms (VITON-HD, DressCode) before claiming commercial freedom.

"""
