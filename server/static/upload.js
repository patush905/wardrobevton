/*
  The three-step upload. The one place plain JS beats HTMX.

    1. POST /api/uploads            -> { upload_id, url }
    2. PUT the File to that url     -> straight to Supabase Storage
    3. POST /api/avatars or /api/garments with the upload_id

  Check size and dimensions client-side before step 1 - it is free and it
  catches the worst inputs. Real validation (is there a whole person in this
  photo) happens server-side during preprocessing.
*/
