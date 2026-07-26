"""
ResumeParser - A modular resume parsing pipeline.

All stages operate on :class:`models.Document` (pages → blocks → spans).

Modules:
    models.py            - Shared data models (Document, Page, BoundingBox, BlockType, etc.)
    resume_ingestion.py  - File validation, type detection, and routing logic
    extraction.py        - Text extraction with coordinate/font metadata (TODO)
    OCR.py               - Optical Character Recognition for scanned documents (TODO)
    layout.py            - Column detection and reading-order reconstruction (TODO)
    section.py           - Section identification and structured field extraction (TODO)
"""
