# AI Wardrobe

Virtual try-on website. Upload a photo of yourself, photograph your clothes,
see yourself wearing them.

**Stack:** FastAPI · Supabase (Postgres + Storage + Auth) · Jinja2 + HTMX ·
Leffa on Modal

## Layout

    app/            the website and JSON API (no GPU, no torch)
      web/          HTML routes
      templates/    Jinja2 pages
      static/       css + the upload helper
      routers/      JSON API
      services/     business logic
      vton/         provider abstraction: mock, Modal, HF Space
    modal_app/      the GPU service, deployed separately
    eval/           the evaluation harness
    tests/

Two deploys, one repo. `app/` is small and boots in seconds; `modal_app/` is
gigabytes of CUDA and only runs while a generation is in flight. They meet at
one HTTPS call.

## Setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e '.[dev]'
    cp .env.example .env          # fill in your Supabase values
    uvicorn app.main:app --reload

Then paste `schema.sql` into the Supabase SQL editor.

## Build order

1. `app/vton/mock.py` and the job pipeline — no GPU needed
2. The website against the mock provider, end to end
3. `eval/` — 100 pairs through a real model, scored
4. `modal_app/` — Leffa and preprocessing on Modal
5. Switch `VTON_PROVIDER` to `modal_leffa`

Step 3 before step 4 is deliberate: find out whether the output is good enough
before investing days in deployment.

## Note

Every file here is a stub containing only a description of what belongs in it.
Nothing runs yet.
