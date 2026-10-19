# pdf-normalizer  

It's just that: a PDF normalizer. I've had course instructors provide scans of excerpts, but these aren't always in an ideal format. This fixes that by cropping, rotating, removing paper tint, dropping blank pages, adding a searchable OCR text layer, and setting PDF page labels to match printed folio numbers. Already-digital pages (real text layer) are passed through untouched instead of being rasterized.

To run it, use `docker compose run normalizer`

By default every `*.pdf` in `pdfs/` is processed. To pick specific files, pass them as arguments (paths are relative to `/app` in the container):

```
docker compose run normalizer python normalize_pdfs.py pdfs/lecture3.pdf pdfs/week2.pdf
```

Every feature is on by default and can be turned off individually, e.g.:

```
docker compose run normalizer python normalize_pdfs.py --no-ocr --no-page-labels
```

Flags: `--no-rotate`, `--no-crop`, `--no-blank-removal`, `--no-ocr`, `--no-page-labels`, `--no-tint-removal` (flatten paper tint / uneven background to white on scanned pages), `--force-rasterize` (rasterize even already-digital pages).

<details>
<summary>Known limitations</summary>

- Tint removal washes out bright areas of photos/maps; use `--no-tint-removal` for those pages.
- Dark-bed scans get speckle noise in the margins and are treated as light-background afterward.
- Scanned pages are saved as JPEG (quality 88), lossy even with `--no-tint-removal`.

</details>
