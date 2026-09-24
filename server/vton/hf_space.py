"""
Fallback: the public Hugging Face Space, via gradio_client.

Free within a daily GPU quota, shared queue, no uptime guarantee. Useful while
Modal is not deployed yet, and as a fallback when it is down.

gradio_client is synchronous - run it in a thread. And verify the Space's
call signature with client.view_api() before relying on it; authors change it
and a stale call comes back as a 422.

"""
