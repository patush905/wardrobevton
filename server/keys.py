"""
Every storage path in one place. Pure functions, no dependencies.

  users/{user_id}/uploads/{upload_id}
  users/{user_id}/avatars/{avatar_id}/original.jpg
  users/{user_id}/garments/{garment_id}/original.jpg
  users/{user_id}/garments/{garment_id}/isolated.png
  users/{user_id}/tryons/{job_id}/result.jpg
  users/{user_id}/tryons/{job_id}/manifest.json

The users/{user_id}/ prefix on everything makes "delete this user's data" a
single prefix removal.

"""
