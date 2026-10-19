# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pymupdf",
#   "Pillow",
#   "numpy",
#   "pytesseract",
#   "opencv-python-headless",
# ]
# ///
"""
Normalize scanned book PDFs:
  - Pages that already carry a real text layer (born-digital, or already OCR'd)
    are passed through untouched -- no rasterization, no quality loss.
  - Remaining pages are rendered, auto-rotated, cropped tightly to content,
    and OCR'd so the output has a searchable text layer (works even over
    diagrams/figures -- sparse-text OCR picks up whatever labels exist and
    degrades gracefully to zero text on pages with none).
  - Blank pages are dropped.
  - Printed folio numbers are detected per page and written as PDF page
    labels, so the output's page counter matches the book's own numbering
    even after blank pages are removed.

Two scan types handled automatically:
  - Dark-background scans (book on dark scanner bed, open two-page spreads)
  - Light-background scans (close-up single pages on white/cream paper)

Most thresholds are derived per page/document rather than hardcoded (Otsu
background/foreground split, border-vs-interior bed detection, native-DPI
render resolution, mm-based crop margin, ink-band folio detection) so the
same script adapts to scans and digital PDFs without per-batch tuning.

Every feature below is on by default and can be individually turned off:
  --no-rotate           don't auto-rotate pages
  --no-crop              don't crop to content
  --no-blank-removal      don't drop blank pages
  --no-ocr                 don't add an OCR text layer to rasterized pages
  --no-page-labels          don't detect folios / set PDF page labels
  --no-tint-removal          don't flatten paper tint / uneven background to white
  --force-rasterize          rasterize every page, even already-digital ones

Run with:  docker compose run normalizer -- [flags]
       or: uv run normalize_pdfs.py [flags]  (requires system Tesseract)
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import fitz  # pymupdf
import numpy as np
import pytesseract
from PIL import Image

PDFS_DIR = Path("pdfs")
OUTPUT_DIR = Path("pdfs_normalized")

# Render resolution is derived from the source image's native DPI (see
# native_dpi()); these bound it to a sane range instead of a fixed value.
DEFAULT_DPI = 300   # used only when no embedded image is found to measure
MIN_RENDER_DPI = 150
MAX_RENDER_DPI = 400
ANALYSIS_DPI_CAP = 200  # OSD accuracy falls off well above this; no need to pay for it
ANALYSIS_DPI_FLOOR = 150  # ...and falls off badly below this

CROP_MARGIN_MM = 3.5  # physical margin kept around detected content, independent of DPI
BLANK_THRESHOLD = 0.01  # absolute floor: fraction of ink pixels below which a page counts blank
BG_BORDER_GAP = 40  # gray-level gap between border and interior median that flags a dark scanner bed

JPEG_QUALITY = 88  # page images embedded in the output PDF

# Tint removal: background is estimated on a downscaled copy (cheap, and the
# background is low-frequency anyway), then divided out of the full-res page.
TINT_SCALE = 0.25
TINT_CLOSE_KERNEL = 61  # at TINT_SCALE; wider than any text stroke or gap, so ink vanishes
TINT_BLUR_SIGMA = 15
TINT_BLACK_POINT = 110  # contrast curve: normalized gray <= this -> black
TINT_WHITE_POINT = 235  # ...and >= this -> white
TINT_GAMMA = 1.3

PAGE_NUM_MAX_DIGITS = 4 # domain fact (no book has a 5-digit printed folio), not a tuning knob
DIGITAL_TEXT_MIN_CHARS = 20  # non-trivial text extracted directly = treat page as already-digital


@dataclass
class Options:
    rotate: bool = True
    crop: bool = True
    remove_blank: bool = True
    ocr: bool = True
    page_labels: bool = True
    tint_removal: bool = True
    force_rasterize: bool = False


def parse_args(argv: list[str] | None = None) -> tuple[Options, list[Path]]:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "files", nargs="*", type=Path, metavar="PDF",
        help=f"PDF files to normalize. Defaults to every *.pdf in {PDFS_DIR}/.",
    )
    p.add_argument("--no-rotate", action="store_true", help="Don't auto-rotate pages.")
    p.add_argument("--no-crop", action="store_true", help="Don't crop to content.")
    p.add_argument("--no-blank-removal", action="store_true", help="Don't drop blank pages.")
    p.add_argument("--no-ocr", action="store_true", help="Don't add an OCR text layer to rasterized pages.")
    p.add_argument("--no-page-labels", action="store_true", help="Don't detect folios or set PDF page labels.")
    p.add_argument("--no-tint-removal", action="store_true", help="Don't flatten paper tint/background to white.")
    p.add_argument(
        "--force-rasterize", action="store_true",
        help="Rasterize every page, even ones that already have a text layer.",
    )
    args = p.parse_args(argv)
    opts = Options(
        rotate=not args.no_rotate,
        crop=not args.no_crop,
        remove_blank=not args.no_blank_removal,
        ocr=not args.no_ocr,
        page_labels=not args.no_page_labels,
        tint_removal=not args.no_tint_removal,
        force_rasterize=args.force_rasterize,
    )
    return opts, args.files


# ---------------------------------------------------------------------------
# Core image helpers
# ---------------------------------------------------------------------------

def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def native_dpi(page: fitz.Page) -> float | None:
    """Best-effort native resolution of the page's largest embedded image."""
    best = None
    for xref, *_ in page.get_images(full=True):
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        rect_w_in, rect_h_in = rect.width / 72, rect.height / 72
        if rect_w_in <= 0 or rect_h_in <= 0:
            continue
        try:
            info = page.parent.extract_image(xref)
        except Exception:
            continue
        dpi_est = max(info["width"] / rect_w_in, info["height"] / rect_h_in)
        if best is None or dpi_est > best:
            best = dpi_est
    return best


def choose_render_dpi(page: fitz.Page) -> int:
    dpi = native_dpi(page)
    if dpi is None:
        return DEFAULT_DPI
    return int(min(max(dpi, MIN_RENDER_DPI), MAX_RENDER_DPI))


def choose_analysis_dpi(render_dpi: int) -> int:
    return int(min(max(render_dpi, ANALYSIS_DPI_FLOOR), ANALYSIS_DPI_CAP))


def otsu_threshold(gray: np.ndarray) -> int:
    """Standard Otsu split -- picks the gray level that best separates two
    populations (scanner bed vs. paper, or paper vs. ink) for this image,
    instead of a threshold tuned on a different batch of scans."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    sum_all = float(np.dot(np.arange(256), hist))
    weight_bg = sum_bg = 0.0
    best_thresh, best_var = 0, -1.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var, best_thresh = var_between, t
    return best_thresh


def classify_background(gray: np.ndarray) -> str:
    """Dark-bg scans have a scanner-bed border noticeably darker than the
    page interior; light-bg (close-up) scans don't, regardless of how much
    ink is on the page -- so compare border vs. interior, not a global
    fixed fraction."""
    h, w = gray.shape
    b = max(1, int(min(h, w) * 0.05))
    border_mask = np.ones_like(gray, dtype=bool)
    border_mask[b:h - b, b:w - b] = False
    border_median = np.median(gray[border_mask])
    interior_median = np.median(gray[~border_mask])
    return "dark" if border_median < interior_median - BG_BORDER_GAP else "light"


def content_mask(gray: np.ndarray, bg: str) -> np.ndarray:
    thresh = otsu_threshold(gray)
    return gray > thresh if bg == "dark" else gray < thresh


def is_blank(gray: np.ndarray, bg: str) -> bool:
    return float(content_mask(gray, bg).mean()) < BLANK_THRESHOLD


# ---------------------------------------------------------------------------
# Tint / background removal
# ---------------------------------------------------------------------------

def remove_tint(img: Image.Image) -> Image.Image:
    """Flatten paper tint and uneven lighting to white; returns grayscale.

    The background is estimated with a morphological CLOSE (not a median
    blur): closing only fills dark features narrower than the kernel, so
    text vanishes from the estimate while a large photo's interior is left
    alone.  A median blur converges on the photo's own tone instead, and
    dividing by that turns photos into white halos.  Output stays grayscale
    (no binarizing) so photos and maps remain readable."""
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    small = cv2.resize(
        gray, (max(1, round(w * TINT_SCALE)), max(1, round(h * TINT_SCALE))),
        interpolation=cv2.INTER_AREA,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TINT_CLOSE_KERNEL, TINT_CLOSE_KERNEL))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=TINT_BLUR_SIGMA)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)  # (w, h): cv2 order

    flat = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0, 0, 255)
    t = np.clip((flat - TINT_BLACK_POINT) / (TINT_WHITE_POINT - TINT_BLACK_POINT), 0, 1)
    return Image.fromarray((t ** TINT_GAMMA * 255.0).astype(np.uint8), "L")


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
# Cropping
# ---------------------------------------------------------------------------

def crop_margin_px(dpi: int) -> int:
    return round(CROP_MARGIN_MM / 25.4 * dpi)


def crop_to_content(img: Image.Image, bg: str, dpi: int) -> Image.Image:
    gray = np.array(img.convert("L"))
    mask = content_mask(gray, bg)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    h, w = gray.shape
    margin = crop_margin_px(dpi)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(0, r0 - margin)
    r1 = min(h - 1, r1 + margin)
    c0 = max(0, c0 - margin)
    c1 = min(w - 1, c1 + margin)
    return img.crop((c0, r0, c1 + 1, r1 + 1))


# ---------------------------------------------------------------------------
# Folio (printed page number) detection
# ---------------------------------------------------------------------------

def detect_folio_band(mask: np.ndarray) -> tuple[int, int] | None:
    """Find an isolated band of ink near the bottom of the page, separated
    from the main text block by whitespace -- that's the folio, wherever it
    actually sits, rather than assuming a fixed fraction of the page height."""
    row_has_ink = np.any(mask, axis=1)
    h = mask.shape[0]
    rows = np.where(row_has_ink)[0]
    if rows.size == 0:
        return None
    bottom = int(rows[-1])
    i = bottom
    while i > 0 and row_has_ink[i]:
        i -= 1
    band_top = i + 1
    if band_top < h * 0.5:
        return None  # bottom-most ink is mid-page; no isolated bottom band
    gap = 0
    j = band_top - 1
    while j > 0 and not row_has_ink[j]:
        gap += 1
        j -= 1
    if gap < h * 0.01:
        return None  # not actually separated from the main text block
    return band_top, bottom


def detect_page_number(img: Image.Image, bg: str) -> int | None:
    gray = np.array(img.convert("L"))
    mask = content_mask(gray, bg)
    band = detect_folio_band(mask)
    h, w = gray.shape
    if band:
        r0, r1 = band
        pad = max(2, int((r1 - r0) * 0.3))
        strip = img.crop((0, max(0, r0 - pad), w, min(h, r1 + pad + 1)))
    else:
        strip = img.crop((0, int(h * 0.88), w, h))
    if bg == "dark":
        strip = Image.fromarray(255 - np.array(strip))
    text = pytesseract.image_to_string(
        strip, config="--psm 11 -c tessedit_char_whitelist=0123456789"
    )
    digits = [int(d) for d in re.findall(rf"\d{{1,{PAGE_NUM_MAX_DIGITS}}}", text)]
    if len(digits) == 1:
        return digits[0]
    if len(digits) == 2 and abs(digits[0] - digits[1]) == 1:
        return min(digits)  # two-page spread: report the left-hand folio
    return None


def build_page_labels(numbers: list[int | None]) -> list[dict]:
    """Turn detected folios into PDF /PageLabels ranges.  A PDF page-label
    range can only ever increment by 1 per subsequent page (spec, not a
    limitation of this code) -- so any detected number that doesn't match
    that plain +1 continuation starts a new range, which is exactly what's
    needed for spreads (folio jumps by 2 each page) or skipped/renumbered
    sections."""
    detected = [(i, n) for i, n in enumerate(numbers) if n is not None]
    if not detected:
        return []
    first_idx, first_num = detected[0]
    start_num = max(1, first_num - first_idx)
    labels = [{"startpage": 0, "prefix": "", "style": "D", "firstpagenum": start_num}]
    last_startpage, last_firstnum = 0, start_num
    for idx, num in detected[1:]:
        implied = last_firstnum + (idx - last_startpage)
        if num != implied:
            labels.append({"startpage": idx, "prefix": "", "style": "D", "firstpagenum": num})
            last_startpage, last_firstnum = idx, num
    return labels


# ---------------------------------------------------------------------------
# Page -> single-page PDF
# ---------------------------------------------------------------------------

def page_to_pdf_bytes(img: Image.Image, dpi: int, ocr: bool) -> bytes:
    """Encode the page as JPEG and wrap it in a one-page PDF.  The JPEG has
    to be handed over as a file/stream: given a PIL image, pytesseract (and
    Pillow's PDF writer) re-encode it losslessly, which bloats the output
    several-fold for no visible gain on scans."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    if ocr:
        # image_to_pdf_or_hocr renders an invisible OCR text layer over
        # the page image -- works on figures/diagrams too, just
        # yielding fewer (or zero) recognized words there instead of
        # failing outright.  Tesseract embeds JPEG input as-is.
        with tempfile.TemporaryDirectory() as tmp:
            jpg_path = Path(tmp) / "page.jpg"
            jpg_path.write_bytes(buf.getvalue())
            return pytesseract.image_to_pdf_or_hocr(
                str(jpg_path), extension="pdf", config=f"--dpi {dpi}"
            )
    with fitz.open() as doc:
        page = doc.new_page(width=img.width / dpi * 72, height=img.height / dpi * 72)
        page.insert_image(page.rect, stream=buf.getvalue())
        return doc.tobytes()


# ---------------------------------------------------------------------------
# Per-PDF processing
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, opts: Options) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {pdf_path.name}")
    print(f"{'─' * 60}")

    doc = fitz.open(pdf_path)
    n = len(doc)
    out_doc = fitz.open()
    page_numbers: list[int | None] = []
    kept = 0
    digital_kept = 0

    for i in range(n):
        page = doc[i]
        label = f"  Page {i + 1:>3}/{n}"

        text = page.get_text("text").strip()
        if not opts.force_rasterize and len(text) > DIGITAL_TEXT_MIN_CHARS:
            out_doc.insert_pdf(doc, from_page=i, to_page=i)
            page_numbers.append(None)
            kept += 1
            digital_kept += 1
            print(f"{label}: [digital text, {len(text)} chars — passed through unchanged]")
            continue

        render_dpi = choose_render_dpi(page)
        analysis_dpi = choose_analysis_dpi(render_dpi)
        img_analysis = render_page(page, dpi=analysis_dpi)
        gray = np.array(img_analysis.convert("L"))
        bg = classify_background(gray)
        if opts.remove_blank and is_blank(gray, bg):
            print(f"{label}: [blank {bg}-bg — skipped]")
            continue

        if opts.rotate:
            rotation, _ = detect_rotation(img_analysis, bg)
        else:
            rotation = 0

        img = render_page(page, dpi=render_dpi)
        if rotation:
            img = img.rotate(-rotation, expand=True)  # negative = CW (Tesseract rotate is CW)
            rot_label = f"rotated {rotation}°"
        else:
            rot_label = "upright"
        # Downstream steps see an image whose polarity may differ from the scan's.
        work_bg = bg
        if opts.tint_removal:
            img = remove_tint(img)
            work_bg = "light"  # dark scanner beds are divided out to white too
        if opts.crop:
            img = crop_to_content(img, work_bg, render_dpi)

        page_bytes = page_to_pdf_bytes(img, render_dpi, ocr=opts.ocr)
        with fitz.open(stream=page_bytes, filetype="pdf") as page_src:
            out_doc.insert_pdf(page_src)

        page_num = detect_page_number(img, work_bg) if opts.page_labels else None
        page_numbers.append(page_num)
        kept += 1
        num_label = f", folio {page_num}" if page_num is not None else ""
        print(f"{label}: [{bg}-bg, {render_dpi} DPI] {rot_label}  →  {img.width}×{img.height} px{num_label}")

    doc.close()
    if kept == 0:
        print("  ⚠  No pages remaining — skipping output.")
        out_doc.close()
        return

    if opts.page_labels:
        labels = build_page_labels(page_numbers)
        if labels:
            out_doc.set_page_labels(labels)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / pdf_path.name
    out_doc.save(out_path)
    out_doc.close()
    note = f" ({digital_kept} passed through as-is)" if digital_kept else ""
    print(f"\n  Saved {kept} page(s) → {out_path}{note}")


def main() -> None:
    opts, files = parse_args()
    if files:
        missing = [f for f in files if not f.is_file()]
        if missing:
            print("File(s) not found: " + ", ".join(str(f) for f in missing))
            sys.exit(1)
        pdfs = files
        print(f"Processing {len(pdfs)} specified PDF(s)")
    else:
        pdfs = sorted(PDFS_DIR.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {PDFS_DIR}/")
            sys.exit(1)
        print(f"Found {len(pdfs)} PDF(s) in {PDFS_DIR}/")
    for pdf_path in pdfs:
        process_pdf(pdf_path, opts)
    print(f"\n✓ Done. Output written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
