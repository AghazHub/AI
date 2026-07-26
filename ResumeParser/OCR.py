"""
OCR.py - Optical Character Recognition for scanned documents.

Handles scanned PDFs (``FileType.PDF_SCANNED``) and standalone images
(``FileType.IMAGE``) by:

1. Converting pages to high-resolution images (pdf2image / PyMuPDF)
2. Preprocessing images with OpenCV (grayscale, binarization, deskew)
3. Running Tesseract OCR via pytesseract
4. Returning a :class:`~ResumeParser.models.Document` with confidence scores

Usage:
    from ResumeParser.resume_ingestion import process_upload
    from ResumeParser.OCR import ocr_document

    result = process_upload("path/to/scanned-resume.pdf")
    doc = ocr_document(result)
    print(doc.raw_text[:500])
    for block in doc.pages[0].blocks:
        print(f"  conf={block.confidence:.2f}  {block.text[:60]}")
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

# -------------- Optional imports (graceful degradation) ---------------

try:
    import cv2
    import numpy as np
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    from pdf2image import convert_from_path
    _HAS_PDF2IMAGE = True
except ImportError:
    _HAS_PDF2IMAGE = False

try:
    import pytesseract
    from pytesseract import Output
    # Verify Tesseract is actually installed (pytesseract wraps the binary)
    pytesseract.get_tesseract_version()
    _HAS_TESSERACT = True
except (ImportError, OSError, RuntimeError):
    _HAS_TESSERACT = False

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from ResumeParser.models import (
    BlockType,
    BoundingBox,
    Document,
    Page,
    TextBlock,
    TextSpan,
)
from ResumeParser.resume_ingestion import FileType, UploadResult


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

# Target DPI for OCR — higher values improve accuracy on small text.
_OCR_DPI = 300

# Tesseract config: treat each page as a single block of text (PSM 6).
# PSM 6 works well for most resume layouts (single-column + headers).
_TESS_CONFIG = "--psm 6 --oem 3"

# Minimum confidence threshold — words below this are excluded.
_MIN_CONFIDENCE = 10.0


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def ocr_document(upload: UploadResult) -> Document:
    """
    Run OCR on a scanned PDF or image file.

    Args:
        upload: An :class:`~ResumeParser.resume_ingestion.UploadResult`
            produced by :func:`~ResumeParser.resume_ingestion.process_upload`.

    Returns:
        A :class:`~ResumeParser.models.Document` with extracted text,
        bounding boxes, and confidence scores.  Check ``doc.error`` for
        failure messages.
    """
    if upload.error:
        return Document(pages=[], raw_text="", error=upload.error)

    if upload.file_type not in (FileType.PDF_SCANNED, FileType.IMAGE):
        return Document(
            pages=[], raw_text="",
            error=f"OCR does not support '{upload.file_type.value}' files. "
                  f"Only scanned PDFs and images are supported.",
        )

    if not _HAS_TESSERACT:
        return Document(
            pages=[], raw_text="",
            error=_tesseract_install_guide(),
        )

    if not _HAS_PIL:
        return Document(
            pages=[], raw_text="",
            error="Pillow is required for image processing. "
                  "Install it with: pip install Pillow",
        )

    file_path = upload.file_path
    ext = Path(file_path).suffix.lower()

    # ── Route to correct page-to-image converter ────────────────
    if upload.file_type == FileType.PDF_SCANNED:
        page_images = _pdf_to_images(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"):
        pil_img = PILImage.open(file_path)
        page_images = [pil_img]
    else:
        return Document(
            pages=[], raw_text="",
            error=f"Unsupported image format: '{ext}'",
        )

    if not page_images:
        return Document(pages=[], raw_text="", error="No pages found to OCR.")

    # ── Process each page ───────────────────────────────────────
    pages: list[Page] = []
    all_text_parts: list[str] = []

    for page_num, pil_img in enumerate(page_images):
        page_blocks, page_width, page_height = _ocr_page(pil_img, page_num)
        page_model = Page(
            page_number=page_num,
            width=page_width,
            height=page_height,
            blocks=page_blocks,
        )
        pages.append(page_model)

        page_text = " ".join(
            b.text for b in page_blocks
        )
        all_text_parts.append(page_text)

    raw_text = "\n".join(all_text_parts)
    return Document(pages=pages, raw_text=raw_text, file_path=str(file_path))


# ══════════════════════════════════════════════════════════════════════
# Page → images
# ══════════════════════════════════════════════════════════════════════

def _pdf_to_images(file_path: str | Path) -> list:
    """Convert PDF pages to PIL images, preferring pdf2image with PyMuPDF fallback."""
    if _HAS_PDF2IMAGE:
        try:
            return convert_from_path(
                str(file_path),
                dpi=_OCR_DPI,
                fmt="png",
            )
        except Exception:
            pass

    if _HAS_PYMUPDF:
        try:
            doc = fitz.open(str(file_path))
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                zoom = _OCR_DPI / 72
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix)
                img_data = pix.tobytes("png")
                pil_img = PILImage.open(io.BytesIO(img_data)).copy()
                images.append(pil_img)
            doc.close()
            return images
        except Exception:
            return []

    return []


# ══════════════════════════════════════════════════════════════════════
# Per-page OCR
# ══════════════════════════════════════════════════════════════════════

def _ocr_page(pil_img, page_number: int) -> tuple[list[TextBlock], float, float]:
    """
    Run the full OCR pipeline on a single PIL image page.

    Returns ``(blocks, page_width_pt, page_height_pt)``.
    """
    # ── Step 1: Convert PIL → OpenCV ────────────────────────────
    img_arr = np.array(pil_img)
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        cv_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    else:
        cv_img = img_arr

    orig_h, orig_w = cv_img.shape[:2]

    # ── Step 2: Preprocess ───────────────────────────────────────
    processed = _preprocess(cv_img)

    # ── Step 3: OCR ──────────────────────────────────────────────
    data = pytesseract.image_to_data(
        processed,
        config=_TESS_CONFIG,
        output_type=Output.DICT,
        lang="eng",
    )

    # ── Step 4: Group words into blocks ──────────────────────────
    blocks = _group_words_into_blocks(data, page_number, orig_w, orig_h)

    # Convert page dimensions from pixels to PDF points
    pt = 72.0 / _OCR_DPI
    return blocks, float(orig_w * pt), float(orig_h * pt)


# ══════════════════════════════════════════════════════════════════════
# Image preprocessing
# ══════════════════════════════════════════════════════════════════════

def _preprocess(cv_img: np.ndarray) -> np.ndarray:
    """
    Apply standard OCR preprocessing pipeline:

    1. Grayscale conversion
    2. Binarization (Otsu threshold)
    3. Deskew (rotation correction)
    """
    # Grayscale
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    # Binarize with Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew
    binary = _deskew(binary)

    return binary


def _deskew(img: np.ndarray) -> np.ndarray:
    """Correct slight rotation in a binary image."""
    coords = np.column_stack(np.where(img > 0))
    if len(coords) < 10:
        return img  # too sparse to determine angle

    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only rotate if angle is significant (> 0.5 degrees)
    if abs(angle) < 0.5:
        return img

    h, w = img.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


# ══════════════════════════════════════════════════════════════════════
# Word → Block grouping
# ══════════════════════════════════════════════════════════════════════

def _group_words_into_blocks(
    data: dict,
    page_number: int,
    img_width: int,
    img_height: int,
) -> list[TextBlock]:
    """
    Group Tesseract word-level data into :class:`TextBlock` objects.

    Tesseract returns each word with a bounding box and confidence.
    Words on the same line (similar y-position) are grouped into a
    single TextBlock per line.
    """
    # Convert pixel coordinates to PDF points (1/72 inch)
    _pixels_to_pt = 72.0 / _OCR_DPI

    # Compute a data-driven line threshold: 1.5x the median word height
    word_heights = []
    for i in range(len(data["text"])):
        h = data["height"][i]
        if h > 0:
            word_heights.append(h)
    median_h = np.median(word_heights) if word_heights else 20
    line_threshold = max(median_h * 1.5, 8)

    n = len(data["text"])
    blocks: list[TextBlock] = []
    current_line_words: list[dict] = []
    prev_top = -1

    def _flush():
        """Create a TextBlock from accumulated words."""
        if not current_line_words:
            return

        texts: list[str] = []
        confs: list[float] = []
        spans: list[TextSpan] = []
        min_x, min_y, max_x, max_y = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )

        for w in current_line_words:
            texts.append(w["text"])
            confs.append(w["conf"])

            # Convert pixel → PDF points
            x0 = w["left"] * _pixels_to_pt
            y0 = w["top"] * _pixels_to_pt
            x1 = (w["left"] + w["width"]) * _pixels_to_pt
            y1 = (w["top"] + w["height"]) * _pixels_to_pt

            min_x = min(min_x, x0)
            min_y = min(min_y, y0)
            max_x = max(max_x, x1)
            max_y = max(max_y, y1)

            # Font size in PDF points
            font_size_pt = w["height"] * _pixels_to_pt

            span = TextSpan(
                text=w["text"],
                bbox=BoundingBox(x0=float(x0), y0=float(y0),
                                 x1=float(x1), y1=float(y1)),
                font_name=None,
                font_size=float(font_size_pt),
                confidence=w["conf"] / 100.0,  # Tesseract returns 0-100
            )
            spans.append(span)

        full_text = " ".join(texts)
        avg_conf = (sum(confs) / len(confs)) / 100.0 if confs else 0.0

        block = TextBlock(
            text=full_text,
            bbox=BoundingBox(x0=float(min_x), y0=float(min_y),
                             x1=float(max_x), y1=float(max_y)),
            page_number=page_number,
            spans=spans,
            block_type=BlockType.TEXT,
            style_name=None,
            confidence=avg_conf,
        )
        blocks.append(block)

    for i in range(n):
        text = data["text"][i].strip()
        conf = data["conf"][i]

        if not text or conf < _MIN_CONFIDENCE:
            continue

        left, top, width, height = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        # Skip empty / invalid bounding boxes
        if width <= 0 or height <= 0:
            continue

        word_data = {
            "text": text,
            "conf": conf,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

        # Check if this word starts a new line (y-position changed significantly)
        if prev_top >= 0 and abs(top - prev_top) > line_threshold:
            _flush()
            current_line_words = []

        current_line_words.append(word_data)
        prev_top = top

    _flush()  # last line
    return blocks


# ══════════════════════════════════════════════════════════════════════
# Error messages
# ══════════════════════════════════════════════════════════════════════

def _tesseract_install_guide() -> str:
    """Return a helpful message explaining how to install Tesseract."""
    lines = [
        "Tesseract OCR engine is not installed or not found on your system PATH.",
        "",
        "── Install Tesseract ──────────────────────────────",
        "",
        "  Windows (with winget):",
        "    winget install UB-Mannheim.TesseractOCR",
        "",
        "  Windows (manual download):",
        "    https://github.com/UB-Mannheim/tesseract/wiki",
        "    Download the 64-bit installer and add Tesseract to your PATH.",
        "",
        "  macOS:",
        "    brew install tesseract",
        "",
        "  Linux:",
        "    sudo apt install tesseract-ocr  # Debian/Ubuntu",
        "    sudo dnf install tesseract       # Fedora",
        "",
        "── After installing ──────────────────────────────",
        "  Restart your terminal / Streamlit app.",
        "  Verify with: tesseract --version",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# CLI entry point (quick manual test)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from ResumeParser.resume_ingestion import process_upload

    if len(sys.argv) < 2:
        print("Usage:  python OCR.py <path/to/scanned-resume.pdf>")
        sys.exit(1)

    upload = process_upload(sys.argv[1])
    if upload.error:
        print(f"Upload error: {upload.error}")
        sys.exit(1)

    doc = ocr_document(upload)
    if doc.error:
        print(f"OCR error: {doc.error}")
        sys.exit(1)

    print(f"Pages:   {len(doc.pages)}")
    print(f"Blocks:  {doc.total_blocks}")
    print(f"Chars:   {len(doc.raw_text)}")
    print("─" * 50)

    for page in doc.pages:
        for i, block in enumerate(page.blocks[:15]):
            preview = block.text[:60] + "…" if len(block.text) > 60 else block.text
            print(f"  [conf={block.confidence:.2f}] p{block.page_number} "
                  f"{preview}")

        if len(page.blocks) > 15:
            print(f"  … and {len(page.blocks) - 15} more blocks")
