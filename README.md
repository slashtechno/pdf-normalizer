# pdf-normalizer  

It's just that: a PDF normalizer. I've had course instructors provide scans of excerpts, but these aren't always in an ideal format. This fixes that by cropping, rotating, dropping blank pages, adding a searchable OCR text layer, and setting PDF page labels to match printed folio numbers. Already-digital pages (real text layer) are passed through untouched instead of being rasterized.

To run it, use `docker compose run normalizer`

Every feature is on by default and can be turned off individually, e.g.:

```
docker compose run normalizer python normalize_pdfs.py --no-ocr --no-page-labels
```

Flags: `--no-rotate`, `--no-crop`, `--no-blank-removal`, `--no-ocr`, `--no-page-labels`, `--force-rasterize` (rasterize even already-digital pages).