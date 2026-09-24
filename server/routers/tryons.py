"""
POST /api/tryons        submit -> 202 {id, status: QUEUED}
GET  /api/tryons/{id}   status, and a signed result URL once COMPLETED
GET  /api/tryons        history

202 and not 200: generation takes seconds to a minute, so the client gets a
job id and polls. Accept an Idempotency-Key header so a retried request never
creates a second job.

"""
