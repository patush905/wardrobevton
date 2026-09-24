"""
POST   /api/garments         register an uploaded garment
GET    /api/garments         the wardrobe, filterable by category, paged
DELETE /api/garments/{id}

Every query filters by the user id from the token. There should be no
get-by-id anywhere that could return another user's row.

"""
