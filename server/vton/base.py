"""
The provider abstraction - the most important interface in the project.

Define:
  VtonRequest   person_url, garment_url, category, steps, seed
  VtonResult    image bytes (or a storage path), model_version, duration_ms
  VtonError     code + retryable flag
  VtonProvider  abstract: provider_id, model_version, try_on(request)

URLs rather than bytes: the provider fetches its own inputs from signed URLs,
so megabytes never move through this process.

Every provider below satisfies this, so swapping one is a config change.

"""
