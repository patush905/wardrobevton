"""
Validate and dispatch a try-on. In this order, and the order matters:

  1. Idempotency - a retried request returns the existing job
  2. Ownership and readiness - the user owns the avatar and garment, both READY
  3. Rate limit - cap in-flight jobs per user
  4. Write the job row (QUEUED)
  5. Dispatch to the provider

Nothing here calls a model. This returns in milliseconds, which is what lets
the endpoint answer 202.

"""
