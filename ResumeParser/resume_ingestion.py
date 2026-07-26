"""
resume_ingestion.py - File validation, type detection, and routing logic.

This module handles the initial stage of the resume parsing pipeline:
    1. Validates the uploaded file (exists, not corrupted, allowed type)
    2. Detects the file type (PDF-digital, PDF-scanned, DOCX, Image)
    3. Routes to the appropriate text extraction pipeline
    4. Returns a structured UploadResult with all gathered metadata

Usage:
    From another module:
        result = process_upload("path/to/resume.pdf")
        print(result.file_type)  # FileType.PDF_DIGITAL

    From the command line:
        python resume_ingestion.py path/to/resume.pdf
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────
# Optional imports – the code degrades gracefully if a library is
# missing so you can start with a minimal install and add later.
# ──────────────────────────────────────────────────────────────────────

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    import magic as libmagic
    _HAS_MAGIC = True
except ImportError:
    _HAS_MAGIC = False

try:
    import filetype
    _HAS_FILETYPE = True
except ImportError:
    _HAS_FILETYPE = False


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Maximum allowed file size: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Text-length threshold used when deciding whether a PDF page is
# "digital" (contains real text) or "scanned" (image-only).
_TEXT_LENGTH_THRESHOLD = 20

# If fewer than this fraction of pages have extractable text, the whole
# PDF is considered scanned.
_SCANNED_PAGE_RATIO = 0.2

# Allowed MIME types.
_ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    # PDF
    'application/pdf',
    # Word
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    # Images
    'image/png',
    'image/jpeg',
    'image/tiff',
    'image/bmp',
    'image/x-ms-bmp',
    # Fallback – some servers send PDFs as octet-stream
    'application/octet-stream',
})




# ──────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────

class FileType(Enum):
    """Supported file types for the resume parsing pipeline.

    Values are strings for clean JSON serialization.
    """

    PDF_DIGITAL = "pdf_digital"    # Born-digital PDF with selectable text
    PDF_SCANNED = "pdf_scanned"    # Image-based PDF – requires OCR
    DOCX = "docx"                  # Microsoft Word document (.docx / .doc)
    IMAGE = "image"                # Scanned image (PNG, JPEG, TIFF, BMP)
    UNSUPPORTED = "unsupported"    # Everything else


@dataclass
class UploadResult:
    """
    Structured result produced by :func:`process_upload`.

    Every field is populated; ``error`` is *None* when the upload is
    valid and the file type was successfully identified.
    """

    file_path: Path
    """Absolute or relative path to the file on disk."""

    original_filename: str
    """The name the uploader originally provided (for display)."""

    file_type: FileType
    """Detected file type – drives downstream routing."""

    file_size_bytes: int
    """File size on disk."""

    file_hash: str
    """SHA-256 hex digest for deduplication / caching."""

    page_count: Optional[int] = None
    """Number of pages (PDFs only)."""

    has_selectable_text: Optional[bool] = None
    """*True* for digital PDFs, *False* for scanned, *None* otherwise."""

    detected_mime: Optional[str] = None
    """MIME type reported by ``python-magic`` or ``filetype``."""

    error: Optional[str] = None
    """Human-readable error description when the file is rejected."""

    raw_metadata: dict = field(default_factory=dict)
    """Extra metadata extracted during inspection (e.g. PDF info dict)."""


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def process_upload(
    file_path: str | Path,
    original_filename: Optional[str] = None,
) -> UploadResult:
    """
    Main entry-point for processing an uploaded resume file.

    The function:

    1. Verifies the file exists, is non-empty, and is within the size
       limit.
    2. Detects the file type using a combination of extension heuristics,
       MIME sniffing, and (for PDFs) PyMuPDF introspection.
    3. Returns an :class:`UploadResult` with all gathered metadata so
       downstream modules know exactly what pipeline to invoke.

    Args:
        file_path: Path to the uploaded file.
        original_filename: Optional original name for display purposes.
            Falls back to ``file_path.name`` when *None*.

    Returns:
        An :class:`UploadResult` – **always** returned, even on error.
        Check ``result.error`` before proceeding.
    """
    path = Path(file_path)
    original_name = original_filename or path.name

    # ── Step 1: Basic validation ────────────────────────────────────
    is_valid, error = _validate_file(path)
    if not is_valid:
        return UploadResult(
            file_path=path,
            original_filename=original_name,
            file_type=FileType.UNSUPPORTED,
            file_size_bytes=0,
            file_hash="",
            error=error,
        )

    file_size = path.stat().st_size
    file_hash = _compute_file_hash(path)

    # ── Step 2: Extension fail-fast ────────────────────────────────
    ext = path.suffix.lower()
    if ext not in {'.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}:
        return UploadResult(
            file_path=path,
            original_filename=original_name,
            file_type=FileType.UNSUPPORTED,
            file_size_bytes=file_size,
            file_hash=file_hash,
            error=f"Unsupported file extension: '{ext}'",
        )

    # ── Step 3: MIME sniffing ───────────────────────────────────────
    mime = _detect_mime(path)

    # ── Step 4: Reject by MIME if it contradicts allowed types ─────
    if mime is not None and mime not in _ALLOWED_MIME_TYPES:
        return UploadResult(
            file_path=path,
            original_filename=original_name,
            file_type=FileType.UNSUPPORTED,
            file_size_bytes=file_size,
            file_hash=file_hash,
            error=f"Unsupported file type (MIME: {mime})",
        )

    # ── Step 5: Type-specific deep inspection ───────────────────────
    file_type = FileType.UNSUPPORTED
    metadata: dict = {}
    page_count: Optional[int] = None
    has_selectable_text: Optional[bool] = None

    if ext == '.pdf':
        pdf_info = _get_pdf_info(path)
        page_count = pdf_info.get("page_count")
        metadata["pdf_metadata"] = pdf_info.get("metadata", {})

        # Guard: if PyMuPDF couldn't open the file at all, flag it
        if page_count is None:
            return UploadResult(
                file_path=path,
                original_filename=original_name,
                file_type=FileType.UNSUPPORTED,
                file_size_bytes=file_size,
                file_hash=file_hash,
                detected_mime=mime,
                error="Could not read PDF file (possibly corrupted)",
            )

        if _is_pdf_scanned(path):
            file_type = FileType.PDF_SCANNED
            has_selectable_text = False
        else:
            file_type = FileType.PDF_DIGITAL
            has_selectable_text = True

    elif ext in ('.docx', '.doc'):
        file_type = FileType.DOCX

    else:
        # Must be an image at this point (extension already validated)
        file_type = FileType.IMAGE

    return UploadResult(
        file_path=path,
        original_filename=original_name,
        file_type=file_type,
        file_size_bytes=file_size,
        file_hash=file_hash,
        page_count=page_count,
        has_selectable_text=has_selectable_text,
        detected_mime=mime,
        raw_metadata=metadata,
    )


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _validate_file(file_path: Path) -> tuple[bool, Optional[str]]:
    """
    Returns ``(True, None)`` if the file passes basic checks.

    Checks performed:
    - Existence
    - Not a directory
    - Non-zero size
    - Below ``MAX_FILE_SIZE_BYTES``
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "File is empty"

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return False, f"File too large ({size_mb:.1f} MB). Maximum is {max_mb:.0f} MB"

    return True, None


def _detect_mime(file_path: Path) -> Optional[str]:
    """
    Sniff the MIME type from file content.

    Prefers ``python-magic`` (libmagic bindings) and falls back to the
    pure-Python ``filetype`` library.  Returns *None* when neither
    library is installed or when sniffing fails.
    """
    path_str = str(file_path)

    if _HAS_MAGIC:
        try:
            return libmagic.from_file(path_str, mime=True)
        except Exception:  # noqa: BLE001
            pass

    if _HAS_FILETYPE:
        try:
            kind = filetype.guess(path_str)
            if kind is not None:
                return kind.mime
        except Exception:  # noqa: BLE001
            pass

    return None


def _is_pdf_scanned(file_path: Path) -> bool:
    """
    Determine whether a PDF is image-based (scanned) or born-digital.

    Uses PyMuPDF to count pages with meaningful selectable text.
    A PDF is considered scanned when **fewer than 20 %** of its pages
    contain at least ``_TEXT_LENGTH_THRESHOLD`` characters of text.

    Falls back to *False* (assume digital) when PyMuPDF is not
    installed or inspection fails.
    """
    if not _HAS_PYMUPDF:
        return False  # optimistic default

    try:
        doc = fitz.open(str(file_path))
    except Exception:  # noqa: BLE001
        return False

    total = len(doc)
    if total == 0:
        doc.close()
        return False

    pages_with_text = 0
    for page_num in range(total):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        if len(text) >= _TEXT_LENGTH_THRESHOLD:
            pages_with_text += 1

    doc.close()

    ratio = pages_with_text / total
    return ratio < _SCANNED_PAGE_RATIO


def _get_pdf_info(file_path: Path) -> dict:
    """Extract page count and PDF metadata via PyMuPDF."""
    info: dict = {}
    if not _HAS_PYMUPDF:
        return info

    try:
        doc = fitz.open(str(file_path))
        info["page_count"] = len(doc)
        info["metadata"] = doc.metadata or {}
        doc.close()
    except Exception:  # noqa: BLE001
        pass

    return info


def _compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of the file contents."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ══════════════════════════════════════════════════════════════════════
# CLI entry point (useful for manual testing)
# ══════════════════════════════════════════════════════════════════════

def _summarize(result: UploadResult) -> str:
    """Pretty-print an ``UploadResult`` for debugging or the CLI."""
    icon = {
        FileType.PDF_DIGITAL: "[PDF]",
        FileType.PDF_SCANNED: "[PDF+OCR]",
        FileType.DOCX: "[DOCX]",
        FileType.IMAGE: "[IMG]",
        FileType.UNSUPPORTED: "[ERR]",
    }.get(result.file_type, "[?]")

    lines = [
        f"{icon}  File:        {result.original_filename}",
        f"    Size:        {result.file_size_bytes / 1024:.1f} KB",
        f"    Type:        {result.file_type.name}",
        f"    Hash:        {result.file_hash[:16]}...",
    ]

    if result.page_count is not None:
        lines.append(f"    Pages:       {result.page_count}")
    if result.has_selectable_text is not None:
        label = "Yes" if result.has_selectable_text else "No (needs OCR)"
        lines.append(f"    Selectable:  {label}")
    if result.detected_mime:
        lines.append(f"    MIME:        {result.detected_mime}")
    if result.error:
        lines.append(f"    Error:       {result.error}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:  python resume_ingestion.py <path/to/resume.pdf>")
        print("        python resume_ingestion.py <path/to/resume.docx>")
        print("        python resume_ingestion.py <path/to/resume.png>")
        sys.exit(1)

    upload_path = sys.argv[1]
    result = process_upload(upload_path)
    print(_summarize(result))

    if result.error:
        sys.exit(1)
