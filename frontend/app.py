"""
frontend/app.py - Streamlit frontend for testing the Resume Parser.

Run with:
    streamlit run frontend/app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

# ── Ensure the project root is on sys.path so we can import
#    ResumeParser modules directly ──────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ResumeParser.resume_ingestion import FileType, process_upload, UploadResult
from ResumeParser.extraction import extract_text
from ResumeParser.OCR import ocr_document
from ResumeParser.layout import detect_layout
from ResumeParser.models import BlockType, Document

# ══════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Resume Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# Custom CSS – modern, clean look
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
    /* ── Global ──────────────────────────────────────── */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    /* ── Stat cards ───────────────────────────────────── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.5rem 0;
    }
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: box-shadow 0.2s;
    }
    .stat-card:hover {
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
    }

    /* ── Pipeline flow ────────────────────────────────── */
    .pipeline {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        padding: 1rem 0;
    }
    .pipeline-step {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #475569;
        white-space: nowrap;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    .pipeline-step.active {
        background: #dbeafe;
        border-color: #3b82f6;
        color: #1d4ed8;
    }
    .pipeline-step.done {
        background: #dcfce7;
        border-color: #22c55e;
        color: #15803d;
    }
    .pipeline-arrow {
        color: #94a3b8;
        font-weight: 700;
        font-size: 1rem;
    }

    /* ── Placeholder cards ────────────────────────────── */
    .placeholder-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .placeholder-card h3 {
        color: #64748b;
        margin-bottom: 0.5rem;
    }
    .placeholder-card p {
        color: #94a3b8;
        font-size: 0.9rem;
        max-width: 480px;
        margin: 0 auto;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.75rem;
        margin-top: 1.25rem;
        text-align: left;
    }
    .feature-item {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #334155;
    }
    .feature-item code {
        background: #f1f5f9;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _type_icon(ft: FileType) -> str:
    return {
        FileType.PDF_DIGITAL:   "📄",
        FileType.PDF_SCANNED:   "📄🔍",
        FileType.DOCX:          "📝",
        FileType.IMAGE:         "🖼️",
        FileType.UNSUPPORTED:   "❌",
    }.get(ft, "❓")


def _type_color(ft: FileType) -> str:
    return {
        FileType.PDF_DIGITAL: "#2563eb",
        FileType.PDF_SCANNED: "#7c3aed",
        FileType.DOCX:        "#0891b2",
        FileType.IMAGE:       "#d97706",
        FileType.UNSUPPORTED: "#dc2626",
    }.get(ft, "#6b7280")


def _render_stat(label: str, value: str, col):
    """Render one stat card inside an st.columns slot."""
    col.markdown(
        f"""<div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_pipeline(current_step: int):
    """Render a horizontal pipeline showing the 5 stages."""
    steps = [
        "📤 Upload",
        "📖 Extraction",
        "🔍 OCR",
        "📐 Layout",
        "📋 Sections",
    ]
    html = '<div class="pipeline">'
    for i, label in enumerate(steps):
        cls = "pipeline-step"
        if i < current_step:
            cls += " done"
        elif i == current_step:
            cls += " active"
        html += f'<span class="{cls}">{label}</span>'
        if i < len(steps) - 1:
            html += '<span class="pipeline-arrow">→</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _show_upload_result(result: UploadResult):
    """Display an UploadResult in a rich card layout."""
    icon = _type_icon(result.file_type)
    color = _type_color(result.file_type)

    st.markdown(
        f"""<div style="background:{color}10;border:1px solid {color}40;
        border-radius:14px;padding:1.25rem 1.5rem;margin:1rem 0;">
        <div style="font-size:1.3rem;font-weight:700;color:{color};">
            {icon}  {result.file_type.name.replace('_', ' · ')}
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    kb = result.file_size_bytes / 1024
    _render_stat("Size", f"{kb:.1f} KB", col1)
    _render_stat("Hash", result.file_hash[:12] + "…", col2)

    if result.page_count is not None:
        _render_stat("Pages", str(result.page_count), col3)
    else:
        _render_stat("Pages", "—", col3)

    label = "Yes" if result.has_selectable_text else "No (needs OCR)" if result.has_selectable_text is False else "—"
    _render_stat("Text", label, col4)

    if result.detected_mime:
        st.caption(f"Detected MIME type: `{result.detected_mime}`")
    if result.raw_metadata.get("pdf_metadata"):
        with st.expander("📑 PDF metadata", expanded=False):
            st.json(result.raw_metadata["pdf_metadata"])

    st.markdown("</div>", unsafe_allow_html=True)


def _placeholder_tab(title: str, description: str, features: list[str],
                     filename_hint: str):
    """Render a placeholder tab for a not-yet-implemented stage."""
    st.markdown(
        f"""<div class="placeholder-card">
            <h3>🚧 {title} — Coming Soon</h3>
            <p>{description}</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.info(
        f"**Module:** `ResumeParser/{filename_hint}`  \n"
        "Once implemented, this tab will show a live demo.",
        icon="ℹ️",
    )
    st.markdown("**Planned features:**")
    for f in features:
        st.markdown(f"- {f}")
    st.divider()
    st.caption(
        "💡 This stage is not yet implemented. "
        "Select the **📤 Upload** tab to test what's already built."
    )


# ══════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📄 Resume Parser")
    st.markdown("---")
    st.markdown(
        "**End-to-end pipeline** to parse resumes into structured JSON.\n\n"
        "Upload a resume file and step through each stage of processing:\n\n"
        "1. **Upload** – validate & detect file type\n"
        "2. **Extraction** – extract raw text with coordinates\n"
        "3. **OCR** – optical character recognition for scans\n"
        "4. **Layout** – detect columns & reading order\n"
        "5. **Sections** – identify + extract structured fields\n\n"
        "---\n"
        "### Supported files\n"
        "• PDF (digital & scanned)\n"
        "• DOCX / DOC\n"
        "• PNG, JPG, TIFF, BMP"
    )
    st.markdown("---")
    st.caption("Built with Streamlit · PyMuPDF · Tesseract")

# ══════════════════════════════════════════════════════════════════════
# Main content
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="main-header">📄 Resume Parser Playground</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Upload a resume and test each stage of the parsing pipeline.</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📤 Upload & Type Detection",
    "📖 Text Extraction",
    "🔍 OCR",
    "📐 Layout Detection",
    "📋 Section Parsing",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 – Upload & Type Detection  (fully functional)
# ══════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### Upload a resume file")
    st.markdown(
        "The first step: validate the file, detect the type (PDF, DOCX, Image), "
        "and for PDFs determine if it's born-digital or scanned."
    )

    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=["pdf", "docx", "doc", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
        help="Maximum file size: 10 MB",
    )

    if uploaded_file is not None:
        # Compute a stable signature for the uploaded file so we can detect
        # when a *new* file is uploaded vs just re-rendering the same one.
        file_sig = f"{uploaded_file.name}:{uploaded_file.size}"
        prev_sig = st.session_state.get("_upload_signature")

        if file_sig != prev_sig:
            # ── This is a NEW upload — replace temp file and clear results ──
            old_temp = st.session_state.pop("_temp_file_path", None)
            if old_temp:
                Path(old_temp).unlink(missing_ok=True)

            suffix = Path(uploaded_file.name).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            st.session_state["_temp_file_path"] = tmp_path
            st.session_state["_upload_signature"] = file_sig

            with st.spinner("Analyzing file…"):
                result = process_upload(tmp_path, original_filename=uploaded_file.name)

            st.session_state["upload_result"] = result
            # Only clear downstream results when a genuinely new file is uploaded
            for key in ["extraction_result", "ocr_result", "layout_result", "section_result"]:
                st.session_state.pop(key, None)
        else:
            # Same file as before — just retrieve the cached upload result
            result = st.session_state.get("upload_result")
            if result is None:
                st.warning("Upload state is missing — please re-upload the file.")
                st.stop()

        st.divider()
        st.markdown("### 📊 Analysis Result")

        if result.error:
            st.error(f"**{result.error}**")
        else:
            _show_upload_result(result)

            # ── Pipeline overview ──────────────────────────────
            st.divider()
            st.markdown("### 🔄 Pipeline Position")
            if result.file_type == FileType.PDF_DIGITAL:
                _render_pipeline(0)
                st.success(
                    "✅ **Digital PDF detected** — ready for text extraction. "
                    "Switch to the **📖 Text Extraction** tab to extract text."
                )
            elif result.file_type == FileType.PDF_SCANNED:
                _render_pipeline(0)
                st.warning(
                    "📄🔍 **Scanned PDF detected** — needs OCR before extraction. "
                    "Proceed to the **🔍 OCR** tab (once implemented)."
                )
            elif result.file_type == FileType.DOCX:
                _render_pipeline(0)
                st.success(
                    "📝 **DOCX detected** — ready for text extraction via python-docx. "
                    "Switch to the **📖 Text Extraction** tab to extract text."
                )
            elif result.file_type == FileType.IMAGE:
                _render_pipeline(0)
                st.info(
                    "🖼️ **Image detected** — needs OCR before further processing. "
                    "Proceed to the **🔍 OCR** tab (once implemented)."
                )

        # NOTE: Temp file is NOT deleted here — downstream tabs (extraction)
        # need the file to still exist. It will be cleaned up when a new file
        # is uploaded, or when the session ends.
    else:
        st.info(
            "👆 Upload a resume file above to see the type detection in action.\n\n"
            "Try different formats: a digital PDF, a scanned PDF, a DOCX, or an image.",
            icon="💡",
        )

        # Example flow preview
        st.divider()
        st.markdown("### 🧪 Try with sample files")
        st.markdown(
            "Once **layout.py**, and **section.py** "
            "are implemented, each tab will show live results step by step."
        )
        _render_pipeline(0)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 – Text Extraction  (fully functional)
# ══════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown("### 📖 Text Extraction")
    st.markdown(
        "Extracts text blocks with spatial coordinates, font names, sizes, "
        "and bold/italic flags using PyMuPDF (for PDFs) or python-docx (for DOCX)."
    )

    upload = st.session_state.get("upload_result")

    if upload is None:
        st.info(
            "👆 Start by uploading a file in the **📤 Upload & Type Detection** tab.",
            icon="💡",
        )
    elif upload.error:
        st.error(f"Upload failed: **{upload.error}**")
    elif upload.file_type in (FileType.PDF_SCANNED, FileType.IMAGE):
        st.warning(
            f"⚠️ **{upload.file_type.name.replace('_', ' ')}** files need OCR "
            "before text can be extracted. This tab only handles digital PDFs "
            "and DOCX files."
        )
        _render_pipeline(1)
    else:
        extract_btn = st.button(
            "▶️ Extract Text",
            type="primary",
            use_container_width=True,
        )

        if extract_btn or "extraction_result" in st.session_state:
            if extract_btn:
                with st.spinner("Extracting text…"):
                    result = extract_text(upload)
                    st.session_state["extraction_result"] = result
            else:
                result = st.session_state["extraction_result"]

            if result.error:
                st.error(f"**{result.error}**")
            else:
                # ── Pipeline ────────────────────────────────────
                st.divider()
                st.markdown("### 🔄 Pipeline Position")
                _render_pipeline(1)

                # ── Stats ───────────────────────────────────────
                col1, col2, col3, col4 = st.columns(4)
                all_blocks = [b for p in result.pages for b in p.blocks]
                _render_stat("Blocks", f"{len(all_blocks)}", col1)
                _render_stat("Characters", f"{len(result.raw_text):,}", col2)
                _render_stat("Pages", str(len(result.pages)), col3)

                # Count unique fonts across all pages
                fonts = set()
                for p in result.pages:
                    for b in p.blocks:
                        for s in b.spans:
                            if s.font_name:
                                fonts.add(s.font_name)
                _render_stat("Fonts", str(len(fonts)) if fonts else "—", col4)

                # ── Raw text preview + blocks per page ───────────
                st.divider()
                c_preview, c_blocks = st.columns([3, 2])

                with c_preview:
                    st.markdown("#### 📝 Raw Text")
                    st.text_area(
                        "Extracted text",
                        result.raw_text[:3000]
                        + ("\n\n… (truncated)" if len(result.raw_text) > 3000 else ""),
                        height=400,
                        label_visibility="collapsed",
                    )

                    # Per-page dimensions
                    if len(result.pages) > 0:
                        dims = [
                            f"p{p.page_number}: {p.width:.0f}×{p.height:.0f}pt"
                            for p in result.pages
                        ]
                        st.caption("Page dimensions: " + " · ".join(dims))

                with c_blocks:
                    st.markdown("#### 📦 Blocks by Page")
                    page_to_show = st.selectbox(
                        "Page",
                        options=range(len(result.pages)),
                        format_func=lambda i: f"Page {i} "
                        f"({len(result.pages[i].blocks)} blocks)",
                        label_visibility="collapsed",
                    )

                    page = result.pages[page_to_show]
                    blocks_on_page = page.blocks

                    st.markdown(
                        f"Showing **{min(len(blocks_on_page), 30)}** "
                        f"of **{len(blocks_on_page)}** blocks on page {page_to_show}."
                    )

                    for i, block in enumerate(blocks_on_page[:30]):
                        preview = block.text[:55]
                        if len(block.text) > 55:
                            preview += "…"

                        meta_parts = []
                        if block.style_name:
                            meta_parts.append(block.style_name)
                        if block.spans:
                            s = block.spans[0]
                            if s.font_name:
                                meta_parts.append(s.font_name)
                            if s.font_size:
                                meta_parts.append(f"{s.font_size:.0f}pt")
                            if s.is_bold:
                                meta_parts.append("**B**")

                        meta_str = f" `{' · '.join(meta_parts)}`" if meta_parts else ""

                        btype_icon = {
                            BlockType.TEXT: "📄",
                            BlockType.IMAGE: "🖼️",
                            BlockType.TABLE: "📊",
                        }.get(block.block_type, "📄")

                        with st.expander(
                            f"{btype_icon} Block {i} · p{block.page_number}{meta_str}",
                            expanded=i < 3,
                        ):
                            st.code(block.text, language="text", line_numbers=False)
                            parts = [
                                f"Type: `{block.block_type.name}`",
                                f"Spans: {len(block.spans)}",
                            ]
                            if block.style_name:
                                parts.append(f"Style: `{block.style_name}`")
                            if block.confidence < 1.0:
                                parts.append(f"Conf: {block.confidence:.2f}")
                            if block.bbox.width > 0:
                                parts.append(f"BBox: ({block.bbox.x0:.0f},{block.bbox.y0:.0f})–"
                                             f"({block.bbox.x1:.0f},{block.bbox.y1:.0f})")
                            st.caption(" · ".join(parts))

                    if len(blocks_on_page) > 30:
                        st.caption(f"… and {len(blocks_on_page) - 30} more blocks on this page")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 – OCR  (fully functional)
# ══════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### 🔍 Optical Character Recognition")
    st.markdown(
        "Converts scanned PDFs and images to text using Tesseract OCR with "
        "OpenCV preprocessing (grayscale, Otsu binarization, deskew)."
    )

    upload = st.session_state.get("upload_result")

    if upload is None:
        st.info(
            "👆 Start by uploading a file in the **📤 Upload & Type Detection** tab.",
            icon="💡",
        )
    elif upload.error:
        st.error(f"Upload failed: **{upload.error}**")
    elif upload.file_type in (FileType.PDF_DIGITAL, FileType.DOCX):
        st.info(
            f"**{upload.file_type.name.replace('_', ' ')}** files don't need OCR "
            "— they already have selectable text. Use the **📖 Text Extraction** tab instead."
        )
        _render_pipeline(2)
    else:
        st.markdown(
            f"**File type:** {upload.file_type.name.replace('_', ' ')}  "
            f"·  **Pages:** {upload.page_count or '?'}"
        )

        ocr_btn = st.button(
            "🔍 Run OCR",
            type="primary",
            use_container_width=True,
        )

        if ocr_btn or "ocr_result" in st.session_state:
            if ocr_btn:
                with st.spinner("Running OCR… (this may take a while)"):
                    result = ocr_document(upload)
                    st.session_state["ocr_result"] = result
            else:
                result = st.session_state["ocr_result"]

            if result.error:
                st.error(f"**{result.error}**")
            else:
                # ── Pipeline ────────────────────────────────────
                st.divider()
                st.markdown("### 🔄 Pipeline Position")
                _render_pipeline(2)

                # ── Stats ───────────────────────────────────────
                col1, col2, col3, col4 = st.columns(4)
                all_blocks = [b for p in result.pages for b in p.blocks]
                _render_stat("Blocks", f"{len(all_blocks)}", col1)
                _render_stat("Characters", f"{len(result.raw_text):,}", col2)
                _render_stat("Pages", str(len(result.pages)), col3)

                # Average confidence
                confs = [b.confidence for p in result.pages for b in p.blocks]
                avg_conf = (sum(confs) / len(confs) * 100) if confs else 0
                _render_stat("Avg. Conf.", f"{avg_conf:.0f}%", col4)

                # ── Text + blocks ────────────────────────────────
                st.divider()
                c_preview, c_blocks = st.columns([3, 2])

                with c_preview:
                    st.markdown("#### 📝 OCR Text")
                    st.text_area(
                        "OCR output",
                        result.raw_text[:3000]
                        + ("\n\n… (truncated)" if len(result.raw_text) > 3000 else ""),
                        height=400,
                        label_visibility="collapsed",
                    )

                with c_blocks:
                    st.markdown("#### 📦 Blocks by Page")
                    page_to_show = st.selectbox(
                        "Page",
                        options=range(len(result.pages)),
                        format_func=lambda i: f"Page {i} "
                        f"({len(result.pages[i].blocks)} blocks)",
                        label_visibility="collapsed",
                    )

                    page = result.pages[page_to_show]
                    blocks_on_page = page.blocks

                    st.markdown(
                        f"Showing **{min(len(blocks_on_page), 30)}** "
                        f"of **{len(blocks_on_page)}** blocks."
                    )

                    for i, block in enumerate(blocks_on_page[:30]):
                        preview = block.text[:55]
                        if len(block.text) > 55:
                            preview += "…"

                        conf_str = f" · conf={block.confidence:.0%}"

                        with st.expander(
                            f"📄 Block {i} · p{block.page_number}{conf_str}",
                            expanded=i < 3,
                        ):
                            st.code(block.text, language="text", line_numbers=False)
                            parts = [
                                f"Confidence: {block.confidence:.0%}",
                                f"Spans: {len(block.spans)}",
                            ]
                            if block.bbox.width > 0:
                                parts.append(
                                    f"BBox: ({block.bbox.x0:.0f},{block.bbox.y0:.0f})–"
                                    f"({block.bbox.x1:.0f},{block.bbox.y1:.0f})"
                                )
                            st.caption(" · ".join(parts))

                    if len(blocks_on_page) > 30:
                        st.caption(
                            f"… and {len(blocks_on_page) - 30} more blocks"
                        )

# ══════════════════════════════════════════════════════════════════════
# TAB 4 – Layout Detection  (fully functional)
# ══════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### 📐 Layout Detection")
    st.markdown(
        "Analyses the spatial arrangement of text blocks to detect columns, "
        "reconstruct reading order, and classify regions."
    )

    # Check which upstream result is available
    doc = (st.session_state.get("extraction_result")
           or st.session_state.get("ocr_result"))
    upload = st.session_state.get("upload_result")

    if upload is None:
        st.info(
            "👆 Start by uploading a file in the **📤 Upload & Type Detection** tab,"
            " then run extraction or OCR first.",
            icon="💡",
        )
    elif doc is None:
        st.info(
            "Run **📖 Text Extraction** or **🔍 OCR** first to generate blocks "
            "for layout analysis.",
            icon="💡",
        )
    elif doc.error:
        st.error(f"Upstream error: **{doc.error}**")
    elif not doc.pages or all(len(p.blocks) == 0 for p in doc.pages):
        st.warning("No blocks found to analyse.")
    else:
        layout_btn = st.button(
            "📐 Analyse Layout",
            type="primary",
            use_container_width=True,
        )

        if layout_btn or "layout_result" in st.session_state:
            if layout_btn:
                with st.spinner("Analysing layout…"):
                    result = detect_layout(doc)
                    st.session_state["layout_result"] = result
            else:
                result = st.session_state["layout_result"]

            if result.error:
                st.error(f"**{result.error}**")
            else:
                # ── Pipeline ────────────────────────────────────
                st.divider()
                st.markdown("### 🔄 Pipeline Position")
                _render_pipeline(3)

                # ── Stats ───────────────────────────────────────
                all_blocks = [b for p in result.pages for b in p.blocks]
                col_count = len({b.column_index for b in all_blocks
                                 if b.column_index is not None})

                col1, col2, col3, col4 = st.columns(4)
                _render_stat("Blocks", f"{len(all_blocks)}", col1)
                _render_stat("Pages", str(len(result.pages)), col2)
                _render_stat("Columns", str(col_count) if col_count > 1 else "1 (single)", col3)

                ordered = sum(1 for b in all_blocks if b.reading_order is not None)
                _render_stat("Ordered", f"{ordered}/{len(all_blocks)}", col4)

                # ── Visual column layout ─────────────────────────
                st.divider()
                c_vis, c_blocks = st.columns([1, 1])

                with c_vis:
                    st.markdown("#### 📊 Column Overview")
                    for page in result.pages:
                        # Group by column
                        cols: dict[int, list[str]] = {}
                        for b in page.blocks:
                            ci = b.column_index or 0
                            cols.setdefault(ci, []).append(b.text[:40])

                        st.markdown(f"**Page {page.page_number}**")
                        if len(cols) <= 1:
                            st.caption("Single column layout")
                            for _, items in cols.items():
                                for item in items[:5]:
                                    st.markdown(f"- {item}…")
                                if len(items) > 5:
                                    st.caption(f"  … {len(items) - 5} more")
                        else:
                            # Multi-column: display side by side
                            col_display = st.columns(len(cols))
                            for (ci, items), col_element in zip(
                                sorted(cols.items()), col_display
                            ):
                                with col_element:
                                    st.markdown(f"> **Col {ci}** ({len(items)})")
                                    for item in items[:6]:
                                        st.caption(f"{item}…")

                with c_blocks:
                    st.markdown("#### 📦 Reading Order")

                    all_sorted = sorted(
                        [b for p in result.pages for b in p.blocks
                         if b.reading_order is not None],
                        key=lambda b: b.reading_order,
                    )

                    st.markdown(
                        f"Showing first **{min(len(all_sorted), 30)}** of "
                        f"**{len(all_sorted)}** ordered blocks."
                    )

                    for i, block in enumerate(all_sorted[:30]):
                        preview = block.text[:50]
                        if len(block.text) > 50:
                            preview += "…"

                        region = ""
                        if block.style_name and "region:" in block.style_name:
                            r = block.style_name.split("region:")[1].split(" |")[0]
                            region = f" · `{r}`"

                        with st.expander(
                            f"#{block.reading_order} · Col {block.column_index} · "
                            f"p{block.page_number}{region}",
                            expanded=i < 5,
                        ):
                            st.code(block.text, language="text", line_numbers=False)
                            parts = [
                                f"Order: #{block.reading_order}",
                                f"Col: {block.column_index}",
                                f"Type: {block.block_type.name}",
                            ]
                            if block.bbox.width > 0:
                                parts.append(
                                    f"Center: ({block.bbox.center_x:.0f},"
                                    f"{block.bbox.center_y:.0f})"
                                )
                            st.caption(" · ".join(parts))

                    if len(all_sorted) > 30:
                        st.caption(f"… and {len(all_sorted) - 30} more ordered blocks")

# ══════════════════════════════════════════════════════════════════════
# TAB 5 – Section Parsing  (placeholder)
# ══════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### 📋 Section Parsing")
    _placeholder_tab(
        title="Section Parsing & Field Extraction",
        description="The final stage: identifies resume sections (Education, Experience, "
                    "Skills, etc.) and extracts structured fields. Uses regex for "
                    "deterministic fields and an LLM (Groq/Cerebras/Mistral) for "
                    "semantically rich sections.",
        features=[
            "🏷️ **Section detection** — regex patterns for known headers + LLM fallback",
            "📋 **Pydantic models** — structured `ExperienceItem`, `EducationItem`, etc.",
            "🤖 **LLM integration** — per-section prompts via Groq / Cerebras / Mistral APIs",
            "✅ **Validation** — date normalization, GPA parsing, deduplication",
            "📦 **Final JSON output** — clean, validated, ready for storage",
        ],
        filename_hint="section.py",
    )

# ══════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════

st.divider()
st.caption(
    "Resume Parser · Built with Streamlit | "
    "[resume_ingestion.py](ResumeParser/resume_ingestion.py) · "
    "[extraction.py](ResumeParser/extraction.py) · "
    "[OCR.py](ResumeParser/OCR.py) · "
    "[layout.py](ResumeParser/layout.py) · "
    "[sections.py](ResumeParser/section.py)"
)
