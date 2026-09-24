"""
Database engine and session factory.

Two Supabase-specific settings that cause intermittent, hard-to-diagnose
failures if you skip them:

  1. Connecting through the transaction pooler (port 6543) means prepared
     statements are not safe - pass statement_cache_size=0 and
     prepared_statement_cache_size=0 in connect_args. Without this you get
     random "prepared statement does not exist" errors under load.
  2. Use NullPool. Supavisor already pools; a second pool here just holds
     connections against your project's limit.

Expose an async get_session() dependency that yields one session per request.

"""
