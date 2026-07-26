"""
layout.py - Column detection, reading-order reconstruction, and region
classification.

Takes a :class:`~ResumeParser.models.Document` produced by extraction or OCR
and enriches each :class:`TextBlock` with::

    block.column_index   — 0-based column number (``None`` before layout)
    block.reading_order  — reading order index across the page
    block.block_type     — may be updated to reflect detected structure

The pipeline is:

1. **Column detection** — cluster blocks by ``center_x`` to find columns
2. **Reading order** — sort left-to-right across columns, top-to-bottom within
3. **Region classification** — detect headers, footers, sidebars based on
   position and page dimensions

Usage:
    from ResumeParser.extraction import extract_text
    from ResumeParser.layout import detect_layout

    doc = extract_text(upload)
    doc = detect_layout(doc)
    for block in doc.pages[0].blocks:
        print(f"col={block.column_index} order={block.reading_order}")
"""

from ResumeParser.models import BlockType, Document, Page, TextBlock


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

# Threshold: blocks whose center_x are within this fraction of page width
# are considered part of the same column.
_COLUMN_GAP_THRESHOLD = 0.08  # 8% of page width

# Minimum blocks required to form a column (ignoring noise).
_MIN_BLOCKS_PER_COLUMN = 2

# Regions: top/bottom fraction of page height considered header/footer.
_HEADER_FOOTER_MARGIN = 0.08  # 8% of page height

# Maximum width fraction for a column to be considered a "sidebar".
_SIDEBAR_MAX_WIDTH = 0.35  # 35% of page width


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def detect_layout(doc: Document) -> Document:
    """
    Analyse the spatial layout of each page in *doc* and enrich every
    :class:`TextBlock` with ``column_index``, ``reading_order``, and
    optionally updated ``block_type``.

    Args:
        doc: A :class:`~ResumeParser.models.Document` with at least one page
             of extracted / OCR'd blocks.

    Returns:
        The same ``Document`` instance (mutated in place) for convenience.
    """
    if doc.error:
        return doc

    if not doc.pages:
        doc.error = "No pages to analyse."
        return doc

    for page in doc.pages:
        _analyse_page(page)

    return doc


# ══════════════════════════════════════════════════════════════════════
# Per-page analysis
# ══════════════════════════════════════════════════════════════════════

def _analyse_page(page: Page) -> None:
    """Run all layout analyses on a single page."""
    if not page.blocks:
        return

    page_width = page.width if page.width > 0 else 612.0
    page_height = page.height if page.height > 0 else 792.0

    # ── 1. Detect columns via x-coordinate clustering ─────────────
    columns = _detect_columns(page.blocks, page_width)
    _assign_columns(page.blocks, columns)

    # ── 2. Assign reading order ───────────────────────────────────
    _assign_reading_order(page.blocks, columns)

    # ── 3. Classify regions (headers, footers, sidebars) ──────────
    _classify_regions(page.blocks, page_width, page_height)


# ══════════════════════════════════════════════════════════════════════
# Column detection
# ══════════════════════════════════════════════════════════════════════

def _detect_columns(
    blocks: list[TextBlock], page_width: float
) -> list[list[TextBlock]]:
    """
    Group blocks into columns by clustering their horizontal centres.

    Uses a simple greedy approach:
    1. Sort blocks by ``center_x``
    2. Start a new column when the x-gap exceeds ``_COLUMN_GAP_THRESHOLD``
       of ``page_width``
    3. Discard columns with fewer than ``_MIN_BLOCKS_PER_COLUMN`` blocks
       as noise (merge them into the nearest column)
    """
    if not blocks:
        return []

    threshold = page_width * _COLUMN_GAP_THRESHOLD

    # Filter out blocks with no meaningful x position (e.g. DOCX null bboxes)
    valid = [b for b in blocks if b.bbox.width > 0 or b.bbox.height > 0]
    if not valid:
        # All blocks have null bboxes — treat as single column
        return [blocks]

    # Sort by horizontal centre
    sorted_blocks = sorted(valid, key=lambda b: b.bbox.center_x)

    # Greedy column grouping
    columns: list[list[TextBlock]] = []
    current_col: list[TextBlock] = [sorted_blocks[0]]
    current_center = sorted_blocks[0].bbox.center_x

    for block in sorted_blocks[1:]:
        gap = abs(block.bbox.center_x - current_center)
        if gap > threshold:
            columns.append(current_col)
            current_col = [block]
            current_center = block.bbox.center_x
        else:
            current_col.append(block)
            # Recompute centre as average of all blocks in this column
            centers = [b.bbox.center_x for b in current_col]
            current_center = sum(centers) / len(centers)

    columns.append(current_col)

    # Merge tiny columns (noise) into a neighbouring column
    columns = _merge_noise_columns(columns)

    # Sort columns left to right by their average x position
    columns.sort(key=lambda col: _column_avg_x(col))

    return columns


def _merge_noise_columns(
    columns: list[list[TextBlock]],
) -> list[list[TextBlock]]:
    """Merge columns with fewer than ``_MIN_BLOCKS_PER_COLUMN`` blocks."""
    if len(columns) < 2:
        return columns

    result: list[list[TextBlock]] = []
    for col in columns:
        if len(col) >= _MIN_BLOCKS_PER_COLUMN:
            result.append(col)
        else:
            # Merge into the closest column by x-distance
            col_avg_x = _column_avg_x(col)
            best_dist = float("inf")
            best_idx = -1
            for i, existing in enumerate(result):
                dist = abs(_column_avg_x(existing) - col_avg_x)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0:
                result[best_idx].extend(col)

    # If everything got merged, return at least one column
    return result or [sum(columns, [])]


def _column_avg_x(blocks: list[TextBlock]) -> float:
    """Average ``center_x`` of all blocks in a column."""
    if not blocks:
        return 0.0
    return sum(b.bbox.center_x for b in blocks) / len(blocks)


# ══════════════════════════════════════════════════════════════════════
# Assign column indices
# ══════════════════════════════════════════════════════════════════════

def _assign_columns(
    blocks: list[TextBlock], columns: list[list[TextBlock]]
) -> None:
    """Set ``column_index`` on each block based on the column it belongs to."""
    # Build a reverse map: block_id → column_index
    col_map: dict[int, int] = {}
    for col_idx, col_blocks in enumerate(columns):
        for b in col_blocks:
            col_map[id(b)] = col_idx

    for b in blocks:
        b.column_index = col_map.get(id(b), 0)


# ══════════════════════════════════════════════════════════════════════
# Reading order
# ══════════════════════════════════════════════════════════════════════

def _assign_reading_order(
    blocks: list[TextBlock], columns: list[list[TextBlock]]
) -> None:
    """
    Assign ``reading_order``: left-to-right across columns, then
    top-to-bottom within each column.
    """
    order = 0
    for col in columns:
        # Sort by y-position (top to bottom), then x (left to right)
        sorted_col = sorted(col, key=lambda b: (b.bbox.center_y, b.bbox.center_x))
        for b in sorted_col:
            b.reading_order = order
            order += 1

    # Blocks not in any column get a high order number
    for b in blocks:
        if b.reading_order is None:
            b.reading_order = order
            order += 1


# ══════════════════════════════════════════════════════════════════════
# Region classification
# ══════════════════════════════════════════════════════════════════════

def _classify_regions(
    blocks: list[TextBlock],
    page_width: float,
    page_height: float,
) -> None:
    """
    Classify blocks by their position on the page.

    Updates ``style_name`` with region hints:
    - Top ``_HEADER_FOOTER_MARGIN`` → ``region:header``
    - Bottom ``_HEADER_FOOTER_MARGIN`` → ``region:footer``
    - Columns narrower than ``_SIDEBAR_MAX_WIDTH`` → ``region:sidebar``
    - Detection is lightweight — detailed heading/section classification
      is deferred to the section parser.
    """
    header_zone = page_height * _HEADER_FOOTER_MARGIN
    footer_zone = page_height * (1.0 - _HEADER_FOOTER_MARGIN)
    sidebar_width_threshold = page_width * _SIDEBAR_MAX_WIDTH

    # Group blocks by column to check column-level width
    col_widths: dict[int, float] = {}
    for b in blocks:
        ci = b.column_index or 0
        col_widths[ci] = max(col_widths.get(ci, 0), b.bbox.width)

    for b in blocks:
        cy = b.bbox.center_y
        ci = b.column_index or 0

        # Header / footer detection by y-position
        if cy < header_zone:
            _mark_region(b, "header")
        elif cy > footer_zone:
            _mark_region(b, "footer")
        # Sidebar detection: narrow column that's not header/footer
        elif col_widths.get(ci, 0) < sidebar_width_threshold:
            _mark_region(b, "sidebar")


def _mark_region(block: TextBlock, region: str) -> None:
    """
    Tag a block with its region type by extending ``block_type`` semantics.

    We store the region name in ``style_name`` (which is otherwise unused
    for PDFs) so that the section parser can consume it without adding a
    new field to ``TextBlock``.
    """
    # Preserve existing style_name from DOCX if present, otherwise set it
    if block.style_name is None or block.style_name.startswith("region:"):
        block.style_name = f"region:{region}"
    else:
        block.style_name = f"{block.style_name} | region:{region}"


# ══════════════════════════════════════════════════════════════════════
# CLI entry point (quick manual test)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from ResumeParser.resume_ingestion import process_upload
    from ResumeParser.extraction import extract_text

    if len(sys.argv) < 2:
        print("Usage:  python layout.py <path/to/resume.pdf>")
        sys.exit(1)

    upload = process_upload(sys.argv[1])
    if upload.error:
        print(f"Upload error: {upload.error}")
        sys.exit(1)

    doc = extract_text(upload)
    if doc.error:
        print(f"Extraction error: {doc.error}")
        sys.exit(1)

    doc = detect_layout(doc)

    print(f"Pages: {len(doc.pages)}\n")

    for page in doc.pages:
        print(f"── Page {page.page_number} ({page.width:.0f} × {page.height:.0f}) ──")

        # Count columns
        col_indices = {b.column_index for b in page.blocks if b.column_index is not None}
        print(f"  Columns detected: {len(col_indices)}")

        for col_idx in sorted(col_indices or {0}):
            col_blocks = [b for b in page.blocks if b.column_index == col_idx]
            col_blocks.sort(key=lambda b: b.reading_order or 0)
            print(f"\n  ── Column {col_idx} ({len(col_blocks)} blocks) ──")
            for b in col_blocks[:8]:
                preview = b.text[:50] + "…" if len(b.text) > 50 else b.text
                region = ""
                if b.style_name and b.style_name.startswith("region:"):
                    region = f" [{b.style_name}]"
                print(f"    #{b.reading_order}  {preview}{region}")
            if len(col_blocks) > 8:
                print(f"    … and {len(col_blocks) - 8} more")
