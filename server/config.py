"""
Settings, read from .env.

Use pydantic-settings BaseSettings so values are validated at startup rather
than blowing up on first use. Cache it with @lru_cache so it is parsed once.

Needs at minimum: DATABASE_URL, SUPABASE_URL, SUPABASE_SERVICE_KEY,
SUPABASE_JWT_SECRET, STORAGE_BUCKET, SIGNED_URL_TTL, VTON_PROVIDER,
MODAL_ENDPOINT_URL, MODAL_CALLBACK_SECRET, DEV_AUTH_BYPASS.

"""
