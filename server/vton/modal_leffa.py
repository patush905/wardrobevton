"""
Calls your deployed Modal endpoint. The real provider.

POST to MODAL_ENDPOINT_URL with the two signed image URLs, the category, steps
and seed. Modal either returns the image, or accepts the job and calls back to
/internal/jobs/{id}/complete - pick one and be consistent.

Expect cold starts. The first request after idle can take 30+ seconds while
the container boots and loads weights. Set a generous timeout and surface a
"warming up" state rather than failing.

Map failures onto VtonError: timeouts and 5xx are retryable, a 4xx from your
own endpoint usually is not.

"""
