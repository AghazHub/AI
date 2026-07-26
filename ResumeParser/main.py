"""
main.py - FastAPI endpoint for the Resume Parser pipeline.

Provides a single ``POST /parse-resume`` endpoint that accepts an uploaded
resume file and optional LLM configuration, runs the full pipeline, and
returns the structured JSON result.

Usage:
    # Start the server
    uvicorn ResumeParser.main:app --reload

    # Test with curl
    curl -X POST http://localhost:8000/parse-resume \\
        -F "file=@resume.pdf" \\
        -F "llm_api_key=gsk_..." \\
        -F "llm_provider=groq"
"""

import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ResumeParser.resume_ingestion import FileType, UploadResult, process_upload

# ── Extraction ────────────────────────────────────────────────────
from ResumeParser.extraction import extract_text

# ── OCR ───────────────────────────────────────────────────────────
from ResumeParser.OCR import ocr_document

# ── Layout ────────────────────────────────────────────────────────
from ResumeParser.layout import detect_layout

# ── Sections ──────────────────────────────────────────────────────
from ResumeParser.section import parse_sections


# ══════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Resume Parser API",
    description="End-to-end resume parsing pipeline: upload a resume and get structured JSON back.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
# Pydantic response models
# ══════════════════════════════════════════════════════════════════════

from pydantic import BaseModel


class FileInfo(BaseModel):
    """Metadata about the uploaded file."""
    filename: str
    file_type: str
    file_size_kb: float
    pages: Optional[int] = None


class ParsingResponse(BaseModel):
    """Top-level API response."""
    success: bool
    file_info: FileInfo
    sections: list[dict] = []
    parsed_resume: Optional[dict] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _upload_result_to_file_info(result: UploadResult) -> FileInfo:
    """Convert an ``UploadResult`` to a ``FileInfo`` response model."""
    return FileInfo(
        filename=result.original_filename,
        file_type=result.file_type.value,
        file_size_kb=result.file_size_bytes / 1024,
        pages=result.page_count,
    )


def _document_to_response(
    doc,
    file_info: FileInfo,
) -> ParsingResponse:
    """Build a ``ParsingResponse`` from a processed ``Document``."""
    if doc.error:
        return ParsingResponse(
            success=False,
            file_info=file_info,
            error=doc.error,
        )

    sections_data = []
    for sec in doc.sections:
        sections_data.append({
            "type": sec.section_type.name.lower(),
            "heading": sec.heading,
        })

    parsed_dict = (
        asdict(doc.parsed_resume)
        if doc.parsed_resume is not None
        else None
    )

    return ParsingResponse(
        success=True,
        file_info=file_info,
        sections=sections_data,
        parsed_resume=parsed_dict,
        raw_text=doc.raw_text[:5000],  # first 5000 chars for preview
    )


# ══════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "Resume Parser API"}


@app.get("/health")
async def health():
    """Health check with version info."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/parse-resume", response_model=ParsingResponse)
async def parse_resume(
    file: UploadFile = File(...),
    llm_api_key: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form("auto"),
    llm_endpoint: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
):
    """
    Upload a resume file and receive fully parsed structured JSON.

    The pipeline runs:
        Upload → Extraction (or OCR) → Layout → Section Parsing

    **Args (form fields):**

    * ``file`` — The resume file (PDF, DOCX, PNG, JPG, TIFF, BMP).
    * ``llm_api_key`` — API key for LLM-powered extraction (optional;
      regex fallback used when omitted).
    * ``llm_provider`` — Provider to use (``\"auto\"``, ``\"groq\"``,
      ``\"openai\"``, ``\"anthropic\"``, ``\"cerebras\"``, ``\"mistral\"``,
      ``\"custom\"``). Default ``\"auto\"``.
    * ``llm_endpoint`` — Override the provider's default API endpoint.
    * ``llm_model`` — Override the provider's default model.

    **Returns:**

    A JSON object with ``success``, ``file_info``, ``sections``,
    ``parsed_resume``, ``raw_text`` (first 5000 chars), and ``error``
    (on failure).
    """
    # ── Validate file ──────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents) / 1024:.1f} KB). Maximum is 10 MB.",
        )

    # Write to a temp file
    suffix = Path(file.filename).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # ── Step 1: Ingestion ─────────────────────────────────────
        upload = process_upload(tmp_path, original_filename=file.filename)
        if upload.error:
            return ParsingResponse(
                success=False,
                file_info=_upload_result_to_file_info(upload),
                error=upload.error,
            )

        file_info = _upload_result_to_file_info(upload)

        # ── Step 2: Extraction or OCR ─────────────────────────────
        if upload.file_type in (FileType.PDF_DIGITAL, FileType.DOCX):
            doc = extract_text(upload)
        elif upload.file_type in (FileType.PDF_SCANNED, FileType.IMAGE):
            doc = ocr_document(upload)
        else:
            return ParsingResponse(
                success=False,
                file_info=file_info,
                error=f"Unsupported file type: {upload.file_type.value}",
            )

        if doc.error:
            return ParsingResponse(
                success=False,
                file_info=file_info,
                error=doc.error,
            )

        # ── Step 3: Layout Analysis ───────────────────────────────
        doc = detect_layout(doc)

        # ── Step 4: Section Parsing ───────────────────────────────
        doc = parse_sections(
            doc,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider if llm_provider != "auto" else None,
            llm_endpoint=llm_endpoint,
            llm_model=llm_model,
        )

        return _document_to_response(doc, file_info)

    except Exception as exc:
        logger.exception("Pipeline failed")
        return ParsingResponse(
            success=False,
            file_info=FileInfo(
                filename=file.filename or "unknown",
                file_type="unknown",
                file_size_kb=0.0,
            ),
            error=f"Pipeline error: {exc}",
        )
    finally:
        # Clean up the temp file
        Path(tmp_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ResumeParser.main:app", host="0.0.0.0", port=8000, reload=True)
