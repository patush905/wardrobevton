"""
The evaluation harness. Build this BEFORE the website, not after.

Take pairs.csv, run each pair through the provider, save the outputs to
results/ alongside a CSV of scores.

Score each result 1-5 on: identity preservation, garment fidelity, pose
preservation, body consistency, texture, realism, artifacts. Tag failure
modes: logo corruption, pattern corruption, sleeve boundary, hands, face
drift, background drift.

The metric that decides whether this project works is garment fidelity - does
the result still look like THAT shirt. Generic realism is not the point.

"""
