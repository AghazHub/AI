"""
extraction.py - Text extraction with spatial and font metadata.

Extracts text from digital PDFs (via PyMuPDF ``page.get_text("dict")``) and
DOCX files (via ``python-docx``), producing a :class:`~ResumeParser.models.Document`
with pages, blocks, and spans.

The public entry-point is :func:`extract_text`, which accepts an
``UploadResult`` and dispatches to the correct backend automatically.

Usage:
    from ResumeParser.resume_ingestion import process_upload
    from ResumeParser.extraction import extract_text

    result = process_upload("path/to/resume.pdf")
    if not result.error and result.has_selectable_text:
        doc = extract_text(result)
        print(f"Extracted {doc.total_blocks} blocks across {len(doc.pages)} pages")
        print(doc.raw_text[:500])
"""

from pathlib import Path
from typing import Optional

# -------------- Optional imports (graceful degradation) ---------------

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    from docx import Document as DocxDocument
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False

from ResumeParser.models import (
    BlockType,
    BoundingBox,
    Color,
    Document,
    Page,
    TextBlock,
    TextSpan,
)
from ResumeParser.resume_ingestion import FileType, UploadResult


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def extract_text(upload: UploadResult) -> Document:
    """
    Extract text from a resume, dispatching to the correct backend based
    on ``upload.file_type``.

    Args:
        upload: An :class:`~ResumeParser.resume_ingestion.UploadResult`
            produced by :func:`~ResumeParser.resume_ingestion.process_upload`.

    Returns:
        A :class:`~ResumeParser.models.Document` with page/block/span
        hierarchy.  Check ``doc.error`` for failure messages.
    """
    if upload.error:
        return Document(pages=[], raw_text="",
                        error=upload.error)

    if upload.file_type == FileType.PDF_DIGITAL:
        if not _HAS_PYMUPDF:
            return Document(
                pages=[], raw_text="",
                error="PyMuPDF is required to extract text from PDFs. "
                      "Install it with: pip install PyMuPDF",
            )
        return _extract_pdf(upload.file_path)

    if upload.file_type == FileType.DOCX:
        if not _HAS_DOCX:
            return Document(
                pages=[], raw_text="",
                error="python-docx is required to extract text from DOCX files. "
                      "Install it with: pip install python-docx",
            )
        return _extract_docx(upload.file_path)

    # Scanned PDFs and images — not yet supported here (see OCR.py).
    return Document(
        pages=[], raw_text="",
        error=f"Cannot extract text from '{upload.file_type.value}' files. "
              f"Run OCR first.",
    )


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _bbox_from_tuple(coords: tuple[float, float, float, float]) -> BoundingBox:
    """Convert a PyMuPDF ``(x0, y0, x1, y1)`` tuple to a :class:`BoundingBox`."""
    return BoundingBox(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])


def _null_bbox() -> BoundingBox:
    """Return a zero-area bounding box used when no spatial data is available."""
    return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)


# ══════════════════════════════════════════════════════════════════════
# PDF extraction — PyMuPDF
# ══════════════════════════════════════════════════════════════════════

def _extract_pdf(file_path: Path) -> Document:
    """Extract text via ``page.get_text("dict")``, building a full Document."""
    doc = fitz.open(str(file_path))
    pages: list[Page] = []
    all_text_parts: list[str] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        rect = page.rect  # (x0, y0, x1, y1) — page dimensions
        page_model = Page(
            page_number=page_num,
            width=rect.width,
            height=rect.height,
        )

        page_dict = page.get_text("dict")

        for raw_block in page_dict.get("blocks", []):
            is_image = raw_block.get("type", 0) == 1

            if is_image:
                bbox = _bbox_from_tuple(raw_block.get("bbox", (0, 0, 0, 0)))
                block = TextBlock(
                    text="[IMAGE]",
                    bbox=bbox,
                    page_number=page_num,
                    spans=[],
                    block_type=BlockType.IMAGE,
                    block_id=raw_block.get("number"),
                )
                page_model.blocks.append(block)
                continue

            # ── Text block ────────────────────────────────────────
            lines = raw_block.get("lines", [])
            if not lines:
                continue

            block_spans: list[TextSpan] = []
            block_text_parts: list[str] = []

            for line in lines:
                for span_data in line.get("spans", []):
                    text = span_data.get("text", "")
                    if not text.strip():
                        continue

                    bbox = _bbox_from_tuple(span_data.get("bbox", (0, 0, 0, 0)))
                    flags = span_data.get("flags", 0)
                    raw_color = span_data.get("color")

                    span = TextSpan(
                        text=text,
                        bbox=bbox,
                        font_name=span_data.get("font"),
                        font_size=span_data.get("size"),
                        is_bold=bool(flags & 2**4),
                        is_italic=bool(flags & 2**1),
                        color=Color(rgb=raw_color) if raw_color is not None else None,
                        confidence=1.0,
                    )
                    block_spans.append(span)
                    block_text_parts.append(text)

            if not block_spans:
                continue

            block_bbox = _bbox_from_tuple(raw_block.get("bbox", (0, 0, 0, 0)))
            full_text = " ".join(block_text_parts)

            block = TextBlock(
                text=full_text,
                bbox=block_bbox,
                page_number=page_num,
                spans=block_spans,
                block_type=BlockType.TEXT,
                # Extraction does NOT classify headings — that's the layout stage's job.
                # Source styles aren't available from raw PDFs; layout analysis will
                # infer structure from font size, bold, position, etc.
                style_name=None,
                block_id=raw_block.get("number"),
                confidence=1.0,
            )
            page_model.blocks.append(block)

        # Collect raw text for this page
        page_text_parts = [
            b.text for b in page_model.blocks if b.block_type != BlockType.IMAGE
        ]
        all_text_parts.extend(page_text_parts)
        pages.append(page_model)

    doc.close()

    raw_text = "\n".join(all_text_parts)
    return Document(pages=pages, raw_text=raw_text, file_path=str(file_path))


# ══════════════════════════════════════════════════════════════════════
# DOCX extraction — python-docx
# ══════════════════════════════════════════════════════════════════════

def _extract_docx(file_path: Path) -> Document:
    """Extract paragraphs and tables via python-docx."""
    doc = DocxDocument(str(file_path))
    blocks: list[TextBlock] = []

    # ── Paragraphs ────────────────────────────────────────────────
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style_name = para.style.name if para.style else None

        spans: list[TextSpan] = []
        for run in para.runs:
            run_text = run.text.strip()
            if not run_text:
                continue

            size_pt: Optional[float] = None
            if run.font.size is not None:
                size_pt = run.font.size.pt

            spans.append(TextSpan(
                text=run_text,
                bbox=_null_bbox(),  # DOCX has no native bounding boxes
                font_name=run.font.name,
                font_size=size_pt,
                is_bold=bool(run.bold),
                is_italic=bool(run.italic),
                confidence=1.0,
            ))

        block = TextBlock(
            text=text,
            bbox=_null_bbox(),
            page_number=0,
            spans=spans or [TextSpan(text=text, bbox=_null_bbox(), confidence=1.0)],
            block_type=BlockType.TEXT,
            # Store the source style name so the layout stage can decide
            # whether this is a heading, body, etc.
            style_name=style_name,
            confidence=1.0,
        )
        blocks.append(block)

    # ── Tables ────────────────────────────────────────────────────
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_text = cell.text.strip()
                if not cell_text:
                    continue
                block = TextBlock(
                    text=cell_text,
                    bbox=_null_bbox(),
                    page_number=0,
                    spans=[TextSpan(text=cell_text, bbox=_null_bbox(),
                                    confidence=1.0)],
                    block_type=BlockType.TABLE,
                    style_name=None,
                    confidence=1.0,
                )
                blocks.append(block)

    raw_text = "\n".join(b.text for b in blocks)
    page = Page(page_number=0, width=0, height=0, blocks=blocks)
    return Document(pages=[page], raw_text=raw_text, file_path=str(file_path))


# ══════════════════════════════════════════════════════════════════════
# CLI entry point (quick manual test)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from ResumeParser.resume_ingestion import process_upload

    if len(sys.argv) < 2:
        print("Usage:  python extraction.py <path/to/resume.pdf>")
        sys.exit(1)

    upload = process_upload(sys.argv[1])
    if upload.error:
        print(f"Upload error: {upload.error}")
        sys.exit(1)

    doc = extract_text(upload)
    if doc.error:
        print(f"Extraction error: {doc.error}")
        sys.exit(1)

    print(f"Pages:   {len(doc.pages)}")
    print(f"Blocks:  {doc.total_blocks}")
    print(f"Chars:   {len(doc.raw_text)}")
    print("─" * 50)

    for page in doc.pages:
        for i, block in enumerate(page.blocks[:20]):
            preview = block.text[:70] + "…" if len(block.text) > 70 else block.text
            meta = []
            if block.style_name:
                meta.append(f"style={block.style_name}")
            if block.spans and block.spans[0].font_name:
                meta.append(block.spans[0].font_name)
            if block.spans and block.spans[0].font_size:
                meta.append(f"{block.spans[0].font_size:.1f}pt")
            meta_str = f"  [{', '.join(meta)}]" if meta else ""
            print(f"  [{block.block_type.name:7s}] p{block.page_number} "
                  f"{preview}{meta_str}")

        if len(page.blocks) > 20:
            print(f"  … and {len(page.blocks) - 20} more blocks on p{page.page_number}")
