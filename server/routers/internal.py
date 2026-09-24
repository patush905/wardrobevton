"""
POST /internal/jobs/{job_id}/complete

Called by the Modal function when a generation finishes. Body carries the
result path, model version and duration; this flips the job row to COMPLETED.

This is a public URL. Require a shared secret header (MODAL_CALLBACK_SECRET)
and compare it with hmac.compare_digest, not ==.

Skip this file entirely if you keep a polling worker instead.

"""
