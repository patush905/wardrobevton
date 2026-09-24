"""
Avatar and garment records.

Creating either one means: confirm the upload landed in storage, copy it from
users/{id}/uploads/ into its permanent prefix, write the row.

Uploads expire; the copy is what makes an image permanent.

"""
