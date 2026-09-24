"""
Supabase Storage.

Two things this exists to guarantee:
  1. Image bytes never pass through the request path. The browser uploads
     straight to storage with a signed URL and downloads the same way.
  2. Leffa on Modal FETCHES its inputs, so signed download URLs are what let
     an external GPU read a private object without holding any credential.

Methods needed: signed_upload_url, signed_download_url, upload_bytes, copy,
exists, remove_prefix.

supabase-py is synchronous. Wrap each call in asyncio.to_thread or you will
block the event loop for every other request in flight.

"""
