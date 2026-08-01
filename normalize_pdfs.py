# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pymupdf",
#   "Pillow",
#   "numpy",
#   "pytesseract",
# ]
# ///
"""
Normalize scanned book PDFs:
  - Auto-rotate pages to correct text orientation
  - Remove blank pages
  - Crop tightly to book content, removing scanner background

Two scan types handled automatically:
  - Dark-background scans (book on dark scanner bed, open two-page spreads)
  - Light-background scans (close-up single pages on white/cream paper)

Run with:  docker compose run normalizer
       or: uv run normalize_pdfs.py  (requires system Tesseract)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import fitz  # pymupdf
import numpy as np
import pytesseract
from PIL import Image

PDFS_DIR = Path("pdfs")
OUTPUT_DIR = Path("pdfs_normalized")
DPI = 300
ANALYSIS_DPI = 200  # higher DPI gives Tesseract more character detail for OSD

CROP_MARGIN_PX = 40
DARK_BG_CONTENT_THRESHOLD = 80
LIGHT_BG_CONTENT_THRESHOLD = 245
BLANK_THRESHOLD = 0.01
DARK_BG_CROP_THRESHOLD = 180


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class PageInfo(NamedTuple):
    blank: bool
    bg: str       # "dark" | "light"
    rotation: int  # CCW degrees to apply (from Tesseract OSD); 0 when blank


# ---------------------------------------------------------------------------
# Core image helpers
# ---------------------------------------------------------------------------

def render_page(page: fitz.Page, dpi: int = DPI) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def classify_background(gray: np.ndarray) -> str:
    """Dark-bg scans have scanner shadow covering 20-40% of pixels below 80 gray.
    Dense essay pages have only ~10-15% below 80 (ink only).
    """
    dark_fraction = float(np.mean(gray < 80))
    return "dark" if dark_fraction > 0.18 else "light"


def content_mask(gray: np.ndarray, bg: str) -> np.ndarray:
    if bg == "dark":
        return gray > DARK_BG_CONTENT_THRESHOLD
    return gray < LIGHT_BG_CONTENT_THRESHOLD


def is_blank(gray: np.ndarray, bg: str) -> bool:
    return float(content_mask(gray, bg).mean()) < BLANK_THRESHOLD


# ---------------------------------------------------------------------------
# Rotation detection via Tesseract OSD
# ---------------------------------------------------------------------------

def detect_rotation(img: Image.Image, bg: str) -> tuple[int, float]:
    """Return (CW degrees to correct, OSD confidence).

    Tesseract OSD is trained on dark-ink-on-white scans.  Dark-bg pages
    (white text on black scanner bed) must be inverted first.
    Returns (0, 0.0) on failure (sparse pages, too little text, etc.)
    """
    if bg == "dark":
        img = Image.fromarray(255 - np.array(img))
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        return int(osd["rotate"]), float(osd["orientation_conf"])
    except pytesseract.TesseractError:
        return 0, 0.0


# ---------------------------------------------------------------------------
# Two-pass PDF processing
# ---------------------------------------------------------------------------

def analyse_pdf(doc: fitz.Document) -> list[PageInfo]:
    """Low-DPI pass: classify every page (blank/bg/rotation)."""
    pages: list[PageInfo] = []
    for i in range(len(doc)):
        img = render_page(doc[i], dpi=ANALYSIS_DPI)
        gray = np.array(img.convert("L"))
        bg = classify_background(gray)
        if is_blank(gray, bg):
            pages.append(PageInfo(blank=True, bg=bg, rotation=0))
            continue
        rot, _ = detect_rotation(img, bg)
        pages.append(PageInfo(blank=False, bg=bg, rotation=rot))
    return pages


def crop_to_content(img: Image.Image, bg: str) -> Image.Image:
    gray = np.array(img.convert("L"))
    if bg == "dark":
        mask = gray > DARK_BG_CROP_THRESHOLD
    else:
        mask = content_mask(gray, bg)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    h, w = gray.shape
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(0, r0 - CROP_MARGIN_PX)
    r1 = min(h - 1, r1 + CROP_MARGIN_PX)
    c0 = max(0, c0 - CROP_MARGIN_PX)
    c1 = min(w - 1, c1 + CROP_MARGIN_PX)
    return img.crop((c0, r0, c1 + 1, r1 + 1))


def process_pdf(pdf_path: Path) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {pdf_path.name}")
    print(f"{'─' * 60}")

    doc = fitz.open(pdf_path)
    n = len(doc)
    print(f"  Analysing {n} pages at {ANALYSIS_DPI} DPI...", flush=True)
    infos = analyse_pdf(doc)

    pages_out: list[Image.Image] = []
    for i, info in enumerate(infos):
        label = f"  Page {i + 1:>3}/{n}"
        if info.blank:
            print(f"{label}: [blank {info.bg}-bg — skipped]")
            continue
        img = render_page(doc[i], dpi=DPI)
        if info.rotation:
            img = img.rotate(-info.rotation, expand=True)  # negative = CW (Tesseract rotate is CW)
            rot_label = f"rotated {info.rotation}°"
        else:
            rot_label = "upright"
        img = crop_to_content(img, info.bg)
        pages_out.append(img)
        print(f"{label}: [{info.bg}-bg] {rot_label}  →  {img.width}×{img.height} px")

    doc.close()
    if not pages_out:
        print("  ⚠  No pages remaining — skipping output.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / pdf_path.name
    pages_out[0].save(
        out_path, format="PDF", save_all=True,
        append_images=pages_out[1:], resolution=DPI,
    )
    print(f"\n  Saved {len(pages_out)} page(s) → {out_path}")


def main() -> None:
    pdfs = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDFS_DIR}/")
        sys.exit(1)
    print(f"Found {len(pdfs)} PDF(s) in {PDFS_DIR}/")
    for pdf_path in pdfs:
        process_pdf(pdf_path)
    print(f"\n✓ Done. Output written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
