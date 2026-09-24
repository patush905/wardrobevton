"""
Pydantic request and response bodies.

One pair per endpoint. Keep these separate from app/models.py - the database
shape and the API shape drift apart, and coupling them means every schema
change is a breaking API change.

Never return storage paths or internal ids the client has no use for; return
signed URLs instead.

"""
