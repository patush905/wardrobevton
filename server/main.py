"""
Application entry point.

Build here:
  - the FastAPI() instance
  - mount app/static at /static
  - include the routers from app/routers/ (the JSON API)
  - include app/web/routes.py (the HTML pages)
  - a /healthz endpoint
  - register the ApiError exception handler from app/errors.py

Run with:  uvicorn app.main:app --reload

If you keep a polling worker instead of the Modal callback, start it from a
lifespan context manager here and cancel it on shutdown.

"""
