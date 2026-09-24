"""
The three tests worth writing first, because each one catches a bug that is
painful to find in production:

  1. submitting twice with the same Idempotency-Key creates ONE job
  2. a user cannot submit a try-on against another user's avatar
  3. a job that fails with a retryable error is requeued, and one that fails
     with a fatal error is not

"""
