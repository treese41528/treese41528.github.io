# Navigator glyphs — upload instructions

Upload the 11 `nav_*.svg` files in this folder to the Supabase bucket at:

    STAT 418 Images / assets / nav /

(so each resolves as .../object/public/STAT%20418%20Images/assets/nav/nav_chapter1.svg etc.)

The dashboard sets `Content-Type: image/svg+xml` automatically from the extension —
required for the CSS backgrounds to render. Until uploaded, the landing-page
navigator simply renders without glyphs (no broken-image icons: they are CSS
backgrounds, not <img> tags). The wiring lives at the end of `_static/custom.css`.
