"""
POST   /api/avatars          register an uploaded photo as the avatar
GET    /api/avatars/{id}     avatar + status + a fresh signed image URL
DELETE /api/avatars/{id}     remove the record and its storage prefix

Mint the image URL per response. Store the path, never the URL.

"""
