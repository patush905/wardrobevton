"""
Shared FastAPI dependencies.

The storage client and the VTON provider are process-wide singletons - build
them once with @lru_cache rather than per request.

"""
