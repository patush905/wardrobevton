"""
SQLAlchemy models. Three tables.

avatars      id, user_id, status, storage_prefix, created_at
garments     id, user_id, status, storage_prefix, category, subcategory,
             attributes (JSONB), attributes_source, created_at
tryon_jobs   id, user_id, avatar_id, garment_ids (JSONB), category,
             status, attempts, provider, model_version, steps, seed,
             modal_call_id, result_path, error_code, error_detail,
             duration_ms, idempotency_key, created_at, started_at,
             completed_at

Two decisions worth keeping:
  - garment_ids is a LIST even though the MVP sends one. Multi-garment later
    then needs no migration.
  - attributes is JSONB, not columns. Classification output keeps changing
    shape; adding a key should not be a migration.

Indexes: (status, created_at) for the job queue, (user_id, created_at) for
history, unique (user_id, idempotency_key).

"""
