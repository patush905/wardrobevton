"""
One-off: pull the Leffa checkpoints from Hugging Face into a Modal Volume.

Run once with `modal run`, not on every container start. Weights in the image
mean every rebuild re-downloads several GB and cold starts get worse; weights
in a Volume are mounted instantly.

franciszzj/Leffa has separate checkpoints for VITON-HD try-on, DressCode
try-on and DeepFashion pose transfer. You want a try-on one.

"""
