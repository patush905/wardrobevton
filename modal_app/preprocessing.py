"""
What the hosted Spaces were doing for you, and now you own.

Leffa needs more than two photos. From the person image, produce:
  - human parsing (SCHP) -> which pixels are hair, face, arms, upper clothes
  - DensePose            -> body surface map
  - an agnostic mask     -> the region to regenerate, with face, hair and
                            hands preserved

This is the fiddliest part of the whole project and the real cost of
self-hosting. Budget more time here than for the Leffa call itself.

Useful side effect: if parsing finds no complete person, reject the job here
and skip generation. That is your photo quality gate now that there is no
on-device check - it costs a couple of GPU-seconds instead of a full run.

"""
