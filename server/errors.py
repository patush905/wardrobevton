"""
One error type, one response shape.

ApiError carries: HTTP status, a stable machine-readable code, and whether
retrying could help. The website switches on `code`, never on message text.

Codes the client needs to handle:
  asset_not_ready (409), rate_limited (429), provider_quota_exhausted (429),
  provider_unavailable (503), generation_failed (502), input_rejected (422)

Plus an exception handler that renders them as JSON.

"""
