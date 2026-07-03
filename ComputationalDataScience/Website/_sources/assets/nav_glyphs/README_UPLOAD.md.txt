# Navigator glyphs — upload instructions

Upload ALL 59 `nav_*.svg` files in this folder to the Supabase bucket at:

    STAT 418 Images / assets / nav /

(each resolves as .../object/public/STAT%20418%20Images/assets/nav/<name>.svg)

- 11 part/chapter/appendix-index glyphs (landing-page Course Content navigator)
- 48 section glyphs (chapter & part index pages + appendices index)

The dashboard sets `Content-Type: image/svg+xml` automatically from the extension —
required for CSS backgrounds to render. Until uploaded, all navigators render
exactly as before (CSS backgrounds: no broken-image icons, screen-reader invisible).
Wiring lives at the end of `_static/custom.css` (href-matched, `.toctree-wrapper`-scoped
so the sidebar never shows glyphs).

NOTE: sphinx incremental builds sometimes skip re-copying _static/custom.css —
if glyphs don't appear after a rebuild, `cp _static/custom.css _build/html/_static/`.
