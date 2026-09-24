"""
Authentication against Supabase Auth.

Supabase issues the JWT when the user signs in, so there is no login endpoint
to write here - only verification. The user id is the token's `sub` claim.

Provide a current_user_id() dependency that:
  - in DEV_AUTH_BYPASS mode, trusts an X-User-Id header (local only)
  - otherwise reads the token from the session cookie OR an Authorization
    header, verifies it, and returns the subject

Check which signing scheme your project uses: a shared HS256 secret, or
asymmetric keys published at /auth/v1/.well-known/jwks.json. They need
different verification code.

"""
