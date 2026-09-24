"""
The website's HTML routes (as opposed to app/routers/, which returns JSON).

  GET  /                landing + sign in
  GET  /avatar          upload your photo
  GET  /wardrobe        garment grid, add garment
  GET  /tryon           pick a garment, submit
  GET  /tryon/{id}      the result page
  GET  /partials/job/{id}   HTML fragment the page polls until COMPLETED
  POST /login, /logout

Return TemplateResponse, not JSON. The polling partial is what makes the
progress UI work without writing any JavaScript.

"""
