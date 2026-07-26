"""
models.py - Shared data models for the resume parsing pipeline.

All stages (extraction, OCR, layout, section parsing) operate on the same
:class:`Document` object, which becomes progressively richer as it moves
through the pipeline::

    Document
    ├── pages : list[Page]
    │   ├── page_number : int
    │   ├── width / height : float
    │   └── blocks : list[TextBlock]
    │       ├── text : str
    │       ├── bbox : BoundingBox
    │       ├── spans : list[TextSpan]
    │       │   ├── text : str
    │       │   ├── bbox : BoundingBox
    │       │   ├── font_name / font_size / is_bold / is_italic
    │       │   └── confidence : float
    │       ├── block_type : BlockType
    │       ├── style_name : str | None   (e.g. "Heading 1" — set by extraction)
    │       └── confidence : float
    └── raw_text : str
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# Primitives
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BoundingBox:
    """
    Axis-aligned bounding rectangle in PDF points (1/72 inch).

    Provides spatial query methods so that downstream stages (layout
    analysis, section detection) can reason about block positions
    without manipulating raw tuples.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    # ── Derived properties ────────────────────────────────────────
    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def center(self) -> tuple[float, float]:
        """``(center_x, center_y)`` — convenient for clustering."""
        return (self.center_x, self.center_y)

    @property
    def area(self) -> float:
        return self.width * self.height

    # ── Spatial queries ───────────────────────────────────────────

    def intersects(self, other: BoundingBox) -> bool:
        """Return ``True`` when the two boxes overlap."""
        return (
            self.x0 < other.x1
            and self.x1 > other.x0
            and self.y0 < other.y1
            and self.y1 > other.y0
        )

    def contains(self, other: BoundingBox) -> bool:
        """Return ``True`` when *other* is fully inside *self*."""
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def distance_to(self, other: BoundingBox) -> float:
        """
        Euclidean distance between the closest edges of two boxes.
        Returns 0 when the boxes overlap.
        """
        if self.intersects(other):
            return 0.0
        dx = max(0.0, self.x0 - other.x1, other.x0 - self.x1)
        dy = max(0.0, self.y0 - other.y1, other.y0 - self.y1)
        return math.hypot(dx, dy)

    def __repr__(self) -> str:
        return f"BoundingBox({self.x0:.1f}, {self.y0:.1f}, {self.x1:.1f}, {self.y1:.1f})"


@dataclass(frozen=True)
class Color:
    """RGB colour represented as an integer (0xRRGGBB)."""

    rgb: int

    @property
    def r(self) -> int:
        return (self.rgb >> 16) & 0xFF

    @property
    def g(self) -> int:
        return (self.rgb >> 8) & 0xFF

    @property
    def b(self) -> int:
        return self.rgb & 0xFF

    @classmethod
    def from_rgb(cls, red: int, green: int, blue: int) -> Color:
        return cls(rgb=(red << 16) | (green << 8) | blue)

    def __repr__(self) -> str:
        return f"Color(#{self.r:02x}{self.g:02x}{self.b:02x})"


# ══════════════════════════════════════════════════════════════════════
# Block-level types
# ══════════════════════════════════════════════════════════════════════

class BlockType(Enum):
    """Semantic category assigned *after* layout analysis (not during extraction).

    Extraction stores the *source style* (e.g. ``style_name="Heading 1"``);
    layout or section analysis promotes that to a ``BlockType``.
    """

    TEXT = auto()
    IMAGE = auto()
    TABLE = auto()


@dataclass
class TextSpan:
    """
    A single text span — the finest unit of text with consistent formatting.

    Corresponds to a PyMuPDF ``span`` (for PDFs) or a ``Run`` (for DOCX).
    """

    text: str
    """Raw text content."""

    bbox: BoundingBox
    """Spatial extent in PDF points."""

    font_name: Optional[str] = None
    """PostScript name of the typeface (e.g. ``'Helvetica'``)."""

    font_size: Optional[float] = None
    """Typeface size in points."""

    is_bold: bool = False
    """``True`` when font flags or run properties indicate bold."""

    is_italic: bool = False
    """``True`` when font flags indicate italic."""

    color: Optional[Color] = None
    """Text colour — extracted from PDF font color or DOCX run colour."""

    confidence: float = 1.0
    """
    Confidence score in ``[0, 1]``.

    = 1.0 for native digital text (perfect confidence).
    < 1.0 for OCR results (set by the OCR stage).
    """

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class TextBlock:
    """
    A cohesive block of text — typically a paragraph, heading, table cell,
    or image placeholder.
    """

    text: str
    """Full concatenated text of the block."""

    bbox: BoundingBox
    """Bounding box encompassing **all** spans in this block."""

    page_number: int
    """Zero-based page index (0 for DOCX files or single-page outputs)."""

    spans: list[TextSpan] = field(default_factory=list)
    """The individual spans that make up this block's content."""

    block_type: BlockType = BlockType.TEXT
    """
    Semantic type assigned by layout analysis.
    Extraction always sets this to ``BlockType.TEXT`` (or ``IMAGE`` / ``TABLE``).
    """

    style_name: Optional[str] = None
    """
    Original style from the source document, e.g. ``'Heading 1'``, ``'Normal'``,
    ``'Title'``.  Set during extraction; consumed by layout analysis.
    """

    block_id: Optional[int] = None
    """Index of this block on its page (for debugging / ordering references)."""

    confidence: float = 1.0
    """Confidence score in ``[0, 1]`` (see :attr:`TextSpan.confidence`)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


# ══════════════════════════════════════════════════════════════════════
# Container types
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Page:
    """
    A single page in the document.

    Introduced so that the pipeline can later add page-level metadata
    (dimensions, rotation, margins, OCR overlays) without changing the
    block-level API.
    """

    page_number: int
    """Zero-based page index."""

    width: float
    """Page width in PDF points (typically 612 for Letter, 595 for A4)."""

    height: float
    """Page height in PDF points."""

    blocks: list[TextBlock] = field(default_factory=list)
    """All text / image blocks extracted from this page."""


@dataclass
class Document:
    """
    Central document model that every pipeline stage operates on.

    Stages progressively enrich this object::

        Extraction  → Document.pages[].blocks[].spans[]  + raw_text
        OCR         → Document.pages[].blocks[].spans[].confidence
        Layout      → Document.pages[].blocks[].block_type
        Sections    → Document.sections (or similar)
    """

    pages: list[Page]
    """Ordered list of pages (not guaranteed to be in reading order — see layout stage)."""

    raw_text: str
    """
    Plain concatenation of ``block.text`` for every block — convenient
    for passing to an LLM without iterating the page/block hierarchy.
    """

    file_path: Optional[str] = None
    """Source file path (for provenance)."""

    error: Optional[str] = None
    """Error message if processing failed at any stage."""

    @property
    def total_blocks(self) -> int:
        return sum(len(p.blocks) for p in self.pages)
