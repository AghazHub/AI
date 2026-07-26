"""
section.py - Resume section identification and structured field extraction.

Identifies resume sections (Education, Experience, Skills, etc.) using regex
patterns for known headers, then extracts structured fields using an LLM
(Groq / Cerebras / Mistral).

Usage:
    from ResumeParser.resume_ingestion import process_upload
    from ResumeParser.extraction import extract_text
    from ResumeParser.layout import detect_layout
    from ResumeParser.section import parse_sections

    upload = process_upload("resume.pdf")
    doc = extract_text(upload)
    doc = detect_layout(doc)
    doc = parse_sections(doc)  # adds doc.sections and doc.parsed_resume

    print(doc.parsed_resume.json(indent=2))
"""

import json
import logging
import re
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

# -------------- Optional imports (graceful degradation) ---------------

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

from ResumeParser.models import (
    BlockType,
    BoundingBox,
    Document,
    EducationItem,
    ExperienceItem,
    ParsedResume,
    ProjectItem,
    ResumeSection,
    SectionType,
    TextBlock,
)


# ══════════════════════════════════════════════════════════════════════
# LLM Provider registry
# ══════════════════════════════════════════════════════════════════════


class LLMFormat(Enum):
    """API format used by the LLM provider."""
    OPENAI = auto()     # Standard chat/completions format (Groq, Cerebras, Mistral, OpenAI)
    ANTHROPIC = auto()  # Anthropic Messages API


@dataclass
class LLMProvider:
    """Configuration for an LLM provider."""
    name: str
    key_prefix: Optional[str]
    endpoint: str
    default_model: str
    api_format: LLMFormat = LLMFormat.OPENAI
    api_key_header: str = "Authorization"
    api_key_template: str = "Bearer {}"
    extra_headers: dict = field(default_factory=dict)


_PROVIDERS: dict[str, LLMProvider] = {
    "groq": LLMProvider(
        name="Groq",
        key_prefix="gsk_",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        default_model="llama-3.1-8b-instant",
    ),
    "openai": LLMProvider(
        name="OpenAI",
        key_prefix="sk-",
        endpoint="https://api.openai.com/v1/chat/completions",
        default_model="gpt-4o-mini",
    ),
    "anthropic": LLMProvider(
        name="Anthropic",
        key_prefix="sk-ant-",
        endpoint="https://api.anthropic.com/v1/messages",
        default_model="claude-3-haiku-20240307",
        api_format=LLMFormat.ANTHROPIC,
        api_key_header="x-api-key",
        api_key_template="{}",
        extra_headers={"anthropic-version": "2023-06-01"},
    ),
    "cerebras": LLMProvider(
        name="Cerebras",
        key_prefix="csk-",
        endpoint="https://api.cerebras.ai/v1/chat/completions",
        default_model="llama3.1-8b",
    ),
    "mistral": LLMProvider(
        name="Mistral",
        key_prefix=None,
        endpoint="https://api.mistral.ai/v1/chat/completions",
        default_model="mistral-small-latest",
    ),
}


def detect_provider(api_key: str) -> LLMProvider:
    """
    Detect the LLM provider from an API key prefix.

    Looks for well-known prefixes (``gsk_`` → Groq, ``sk-`` → OpenAI,
    ``sk-ant-`` → Anthropic, ``csk-`` → Cerebras). Falls back to Groq
    when no prefix is recognised.
    """
    for key, provider in _PROVIDERS.items():
        if provider.key_prefix and api_key.startswith(provider.key_prefix):
            return provider
    # Mistral has no consistent key prefix — check for any non-empty key
    if api_key.strip():
        return _PROVIDERS["mistral"]
    return _PROVIDERS["groq"]


# ══════════════════════════════════════════════════════════════════════
# Section header patterns (regex-based fast path)
# ══════════════════════════════════════════════════════════════════════

# Maps regex patterns → SectionType. Order matters: first match wins.
_SECTION_PATTERNS: list[tuple[re.Pattern, SectionType]] = [
    # Education
    (re.compile(
        r"^\s*(education|academic\s*background|academic\s*qualifications|"
        r"educational\s*background|qualifications)\b",
        re.IGNORECASE,
    ), SectionType.EDUCATION),
    # Experience
    (re.compile(
        r"^\s*(experience|work\s*experience|professional\s*experience|"
        r"employment|employment\s*history|work\s*history|career)\b",
        re.IGNORECASE,
    ), SectionType.EXPERIENCE),
    # Skills
    (re.compile(
        r"^\s*(skills|technical\s*skills|core\s*competencies|expertise|"
        r"competencies|technologies|tech\s*stack|proficiencies)\b",
        re.IGNORECASE,
    ), SectionType.SKILLS),
    # Summary / Profile
    (re.compile(
        r"^\s*(summary|professional\s*summary|profile|about\s*me|"
        r"objective|career\s*objective|personal\s*statement)\b",
        re.IGNORECASE,
    ), SectionType.SUMMARY),
    # Projects
    (re.compile(
        r"^\s*(projects|personal\s*projects|key\s*projects|"
        r"project\s*experience|portfolio)\b",
        re.IGNORECASE,
    ), SectionType.PROJECTS),
    # Certifications
    (re.compile(
        r"^\s*(certifications|certificates|licenses|"
        r"professional\s*certifications)\b",
        re.IGNORECASE,
    ), SectionType.CERTIFICATIONS),
    # Languages
    (re.compile(
        r"^\s*(languages|language\s*proficiency|languages\s*spoken)\b",
        re.IGNORECASE,
    ), SectionType.LANGUAGES),
    # Publications
    (re.compile(
        r"^\s*(publications|papers|research|research\s*publications)\b",
        re.IGNORECASE,
    ), SectionType.PUBLICATIONS),
    # Volunteering
    (re.compile(
        r"^\s*(volunteering|volunteer|volunteer\s*experience|"
        r"community|community\s*service)\b",
        re.IGNORECASE,
    ), SectionType.VOLUNTEERING),
]

# Patterns for deterministic field extraction (no LLM needed)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
)
_LINKEDIN_PATTERN = re.compile(
    r"(linkedin\.com/(?:in|pub)/[a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
_GITHUB_PATTERN = re.compile(
    r"github\.com/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════
# LLM system prompt — full-document extraction (the primary engine)
# ══════════════════════════════════════════════════════════════════════

_FULL_EXTRACT_SYSTEM_PROMPT = """\
You are a precise resume parser. Given the FULL TEXT of a resume below, \
extract ALL structured information.

Return ONLY valid JSON — no explanation, no markdown, no extra text. The \
response MUST be parseable as JSON directly.

The JSON must have this exact structure:
{
  "name": "Full Name or null",
  "email": "email@example.com or null",
  "phone": "+1 234 567 8900 or null",
  "linkedin": "linkedin.com/in/username or null",
  "github": "github.com/username or null",
  "summary": "Professional summary text or null",
  "sections": [
    {"type": "experience", "heading": "Work Experience"},
    {"type": "education", "heading": "Education"},
    {"type": "skills", "heading": "Skills"}
  ],
  "experience": [
    {"company": "...", "role": "...", "start_date": "...", "end_date": "...", "description": ["..."], "technologies": ["..."]}
  ],
  "education": [
    {"institution": "...", "degree": "...", "field": "...", "graduation_date": "...", "gpa": "..."}
  ],
  "skills": ["skill1", "skill2"],
  "certifications": ["cert1"],
  "projects": [
    {"name": "...", "description": "...", "technologies": ["..."]}
  ]
}

Section type must be one of: summary, experience, education, skills, \
projects, certifications, languages, publications, volunteering, custom.

Be thorough. Extract every detail you can find. Use null for missing \
fields. Arrays can be empty."""  # noqa: E501

_ANTHROPIC_FULL_PROMPT = _FULL_EXTRACT_SYSTEM_PROMPT + (
    "\n\nIMPORTANT: You MUST wrap your JSON response in "
    "```json\n...\n``` tags."
)


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════

def parse_sections(
    doc: Document,
    llm_api_key: Optional[str] = None,
    llm_provider: Optional[str] = None,
    llm_endpoint: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Document:
    """
    Identify resume sections and extract structured fields.

    **LLM-first approach** (when an API key is provided):

    1. Regex runs first as a fallback for deterministic fields (email,
       phone, LinkedIn, GitHub).

    2. Then the **entire ``raw_text``** is sent to the LLM in a single
       call. The LLM identifies sections, extracts contact info,
       experience, education, skills, projects, and certifications.

    3. The LLM's non-null values **override** the regex fallback,
       ensuring the LLM is the primary extraction engine.

    4. If the LLM call fails, the regex fallback values are retained.

    **Regex-only fallback** (when no API key is provided):

    - Detect sections via header regex patterns (for UI display).
    - Extract email, phone, LinkedIn, GitHub via regex. Everything else
      remains empty.

    Args:
        doc: A :class:`~ResumeParser.models.Document` with layout analysis
            already applied (reading order, columns, etc.).
        llm_api_key: API key for the LLM provider. When provided, the LLM
            becomes the primary extraction engine. When ``None``, only
            regex-based extraction runs.
        llm_provider: Provider identifier (``"auto"``, ``"groq"``, etc.).
        llm_endpoint: Optional endpoint override.
        llm_model: Optional model override.

    Returns:
        The same ``Document`` with ``sections`` and ``parsed_resume`` set.
    """
    if doc.error:
        return doc

    # ══════════════════════════════════════════════════════════════
    # Step 1 — Regex fallback: deterministic fields
    # ══════════════════════════════════════════════════════════════
    parsed = ParsedResume()
    _extract_deterministic_fields(doc.raw_text, parsed)

    # ══════════════════════════════════════════════════════════════
    # Step 2 — LLM overrides regex values with non-null results
    # ══════════════════════════════════════════════════════════════
    if llm_api_key:
        llm_result = _llm_extract_full_document(
            doc, api_key=llm_api_key,
            provider_name=llm_provider or "auto",
            endpoint_override=llm_endpoint,
            model_override=llm_model,
        )

        if llm_result is not None:
            # LLM non-null scalar fields override regex
            for field in ("name", "email", "phone", "linkedin", "github", "summary"):
                llm_val = getattr(llm_result, field, None)
                if llm_val is not None:
                    setattr(parsed, field, llm_val)
            # LLM list fields override (non-empty lists win)
            if llm_result.experience:
                parsed.experience = llm_result.experience
            if llm_result.education:
                parsed.education = llm_result.education
            if llm_result.skills:
                parsed.skills = llm_result.skills
            if llm_result.certifications:
                parsed.certifications = llm_result.certifications
            if llm_result.projects:
                parsed.projects = llm_result.projects

            doc.parsed_resume = parsed
            return doc

        logger.warning("Full-document LLM extraction failed — regex results retained.")

    # ══════════════════════════════════════════════════════════════
    # Step 3 — Regex-only fallback: detect sections for UI display
    # ══════════════════════════════════════════════════════════════
    doc.sections = _detect_sections(doc)
    doc.parsed_resume = parsed
    return doc


def _llm_extract_full_document(
    doc: Document,
    api_key: str,
    provider_name: str = "auto",
    endpoint_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Optional[ParsedResume]:
    """
    Send the entire resume raw text to the LLM and parse the structured JSON.

    Returns a :class:`ParsedResume` with all fields populated by the LLM,
    or ``None`` if the call fails.

    Also populates ``doc.sections`` with the sections identified by the LLM
    so the UI can display them.
    """
    if not _HAS_HTTPX:
        logger.warning("httpx is not installed — install with: pip install httpx")
        return None

    if not doc.raw_text.strip():
        logger.warning("No raw text to extract from.")
        return None

    # ── Resolve provider ───────────────────────────────────────────
    provider = _resolve_provider(api_key, provider_name, endpoint_override)
    endpoint = endpoint_override or provider.endpoint
    model = model_override or provider.default_model

    if not model:
        logger.warning("No model specified — cannot call LLM.")
        return None

    # ── Build headers ─────────────────────────────────────────────
    headers = {
        provider.api_key_header: provider.api_key_template.format(api_key),
        "Content-Type": "application/json",
    }
    headers.update(provider.extra_headers)

    # ── Select the right system prompt ────────────────────────────
    system_prompt = (
        _ANTHROPIC_FULL_PROMPT
        if provider.api_format == LLMFormat.ANTHROPIC
        else _FULL_EXTRACT_SYSTEM_PROMPT
    )

    user_prompt = f"Extract all structured information from this resume:\n\n{doc.raw_text}"

    with httpx.Client(timeout=90.0) as client:
        result = _llm_call(client, endpoint, headers, model, provider, user_prompt,
                           system_override=system_prompt)

    if result is None:
        return None

    # ── Map the LLM response into ParsedResume ────────────────────
    return _map_llm_response_to_parsed(doc, result)


# ══════════════════════════════════════════════════════════════════════
# Section detection
# ══════════════════════════════════════════════════════════════════════

def _detect_sections(doc: Document) -> list[ResumeSection]:
    """Split blocks into sections by matching header patterns."""
    # Flatten blocks in reading order across all pages
    all_blocks = sorted(
        [b for p in doc.pages for b in p.blocks],
        key=lambda b: (
            b.page_number,
            b.reading_order if b.reading_order is not None else 9999,
            b.bbox.center_y,
        ),
    )

    sections: list[ResumeSection] = []
    current_section: Optional[ResumeSection] = None

    for block in all_blocks:
        if block.block_type is BlockType.IMAGE:
            continue

        text = block.text.strip()
        if not text:
            continue

        # Check if this block is a section header
        matched_type = _match_header(text)

        if matched_type is not None:
            # Start a new section
            logger.debug("Section header matched: '%s' → %s", text[:60], matched_type.name)
            current_section = ResumeSection(
                section_type=matched_type,
                heading=text,
                heading_bbox=block.bbox if block.bbox.width > 0 else None,
            )
            sections.append(current_section)
        elif current_section is not None:
            # Accumulate into current section
            current_section.blocks.append(block)
            current_section.raw_text += ("\n" if current_section.raw_text else "") + text

    if not sections:
        logger.info(
            "No sections detected via regex in %d blocks. "
            "First few block texts: %s",
            len(all_blocks),
            [b.text.strip()[:50] for b in all_blocks[:8]],
        )

    return sections


def _match_header(text: str) -> Optional[SectionType]:
    """Return the ``SectionType`` if *text* matches a known header pattern."""
    for pattern, section_type in _SECTION_PATTERNS:
        if pattern.match(text):
            return section_type
    return None


# ══════════════════════════════════════════════════════════════════════
# Deterministic field extraction
# ══════════════════════════════════════════════════════════════════════

def _extract_deterministic_fields(raw_text: str, parsed: ParsedResume) -> None:
    """Extract email, phone, LinkedIn, GitHub URLs via regex."""
    # Email
    emails = _EMAIL_PATTERN.findall(raw_text)
    if emails:
        parsed.email = emails[0]

    # Phone
    phones = _PHONE_PATTERN.findall(raw_text)
    if phones:
        # Take the longest match (most complete number)
        parsed.phone = max(phones, key=len).strip()

    # LinkedIn
    linkedin = _LINKEDIN_PATTERN.search(raw_text)
    if linkedin:
        parsed.linkedin = linkedin.group(0)

    # GitHub
    github = _GITHUB_PATTERN.search(raw_text)
    if github:
        parsed.github = f"github.com/{github.group(1)}"


# ══════════════════════════════════════════════════════════════════════
# LLM call helpers
# ══════════════════════════════════════════════════════════════════════

# Default per-section prompts — kept as fallback defaults for _llm_call
# when no system_override is provided (currently unused in LLM-first flow).
_SYSTEM_PROMPT = _FULL_EXTRACT_SYSTEM_PROMPT
_ANTHROPIC_SYSTEM_PROMPT = _ANTHROPIC_FULL_PROMPT


def _resolve_provider(
    api_key: str,
    provider_name: str,
    endpoint_override: Optional[str] = None,
) -> LLMProvider:
    """Resolve the :class:`LLMProvider` from user configuration."""
    if provider_name == "custom":
        return LLMProvider(
            name="Custom",
            key_prefix=None,
            endpoint=endpoint_override or "https://api.groq.com/openai/v1/chat/completions",
            default_model="",
        )
    if provider_name == "auto" or provider_name not in _PROVIDERS:
        return detect_provider(api_key)
    return _PROVIDERS[provider_name]


def _llm_call(
    client: "httpx.Client",
    endpoint: str,
    headers: dict,
    model: str,
    provider: LLMProvider,
    prompt: str,
    system_override: Optional[str] = None,
) -> Optional[dict]:
    """
    Make a single LLM API call and parse the JSON response.

    Dispatches to the correct call handler based on the provider's
    :attr:`LLMProvider.api_format`.  *system_override* replaces the
    default system prompt when provided.
    """
    if provider.api_format == LLMFormat.ANTHROPIC:
        sys_prompt = system_override or _ANTHROPIC_SYSTEM_PROMPT
        return _llm_call_anthropic(client, endpoint, headers, model, prompt, sys_prompt)
    sys_prompt = system_override or _SYSTEM_PROMPT
    return _llm_call_openai(client, endpoint, headers, model, prompt, sys_prompt)


def _llm_call_openai(
    client: "httpx.Client",
    endpoint: str,
    headers: dict,
    model: str,
    prompt: str,
    system_prompt: str = _SYSTEM_PROMPT,
) -> Optional[dict]:
    """OpenAI-compatible ``/chat/completions`` call."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as exc:
        logger.warning("LLM call (OpenAI) failed: %s", exc)
        return None


def _llm_call_anthropic(
    client: "httpx.Client",
    endpoint: str,
    headers: dict,
    model: str,
    prompt: str,
    system_prompt: str = _ANTHROPIC_SYSTEM_PROMPT,
) -> Optional[dict]:
    """
    Anthropic ``/v1/messages`` call.

    Anthropic does not support ``response_format``, so we embed the JSON
    constraint in the system prompt and ask for a ```json…``` block.
    """
    payload = {
        "model": model,
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    try:
        resp = client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Anthropic returns content as a list of content blocks
        full_text = "".join(
            block["text"] for block in data.get("content", [])
            if block.get("type") == "text"
        )
        # Strip ```json … ``` fence if present
        json_str = re.sub(
            r"^```(?:json)?\s*", "", full_text, flags=re.MULTILINE
        ).rstrip("`\n ")
        return json.loads(json_str)
    except Exception as exc:
        logger.warning("LLM call (Anthropic) failed: %s", exc)
        return None





# ══════════════════════════════════════════════════════════════════════
# LLM response → ParsedResume mapper (full-document extraction)
# ══════════════════════════════════════════════════════════════════════

_SECTION_TYPE_LABELS = {
    "summary": SectionType.SUMMARY,
    "experience": SectionType.EXPERIENCE,
    "education": SectionType.EDUCATION,
    "skills": SectionType.SKILLS,
    "projects": SectionType.PROJECTS,
    "certifications": SectionType.CERTIFICATIONS,
    "languages": SectionType.LANGUAGES,
    "publications": SectionType.PUBLICATIONS,
    "volunteering": SectionType.VOLUNTEERING,
    "custom": SectionType.CUSTOM,
}


def _map_llm_response_to_parsed(doc: Document, result: dict) -> ParsedResume:
    """Map the LLM's full-document JSON response into a ``ParsedResume``."""
    parsed = ParsedResume(
        name=result.get("name"),
        email=result.get("email"),
        phone=result.get("phone"),
        linkedin=result.get("linkedin"),
        github=result.get("github"),
        summary=result.get("summary"),
    )

    # ── Experience ────────────────────────────────────────────────
    for item in result.get("experience", []):
        safe = {k: item[k] for k in ("company", "role", "start_date", "end_date",
                                     "description", "technologies") if k in item}
        parsed.experience.append(ExperienceItem(**safe))

    # ── Education ─────────────────────────────────────────────────
    for item in result.get("education", []):
        safe = {k: item[k] for k in ("institution", "degree", "field",
                                     "graduation_date", "gpa") if k in item}
        parsed.education.append(EducationItem(**safe))

    # ── Skills ────────────────────────────────────────────────────
    parsed.skills = result.get("skills", [])

    # ── Certifications ────────────────────────────────────────────
    parsed.certifications = result.get("certifications", [])

    # ── Projects ──────────────────────────────────────────────────
    for item in result.get("projects", []):
        safe = {k: item[k] for k in ("name", "description", "technologies") if k in item}
        parsed.projects.append(ProjectItem(**safe))

    # ── Sections (populated for UI display) ───────────────────────
    sections = []
    for sec in result.get("sections", []):
        stype = _SECTION_TYPE_LABELS.get(
            sec.get("type", "").lower(), SectionType.CUSTOM
        )
        sections.append(ResumeSection(
            section_type=stype,
            heading=sec.get("heading", ""),
        ))
    doc.sections = sections

    return parsed


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from ResumeParser.resume_ingestion import process_upload
    from ResumeParser.extraction import extract_text
    from ResumeParser.layout import detect_layout

    if len(sys.argv) < 2:
        print("Usage:  python section.py <path/to/resume.pdf> [api_key]")
        sys.exit(1)

    api_key = sys.argv[2] if len(sys.argv) > 2 else None

    upload = process_upload(sys.argv[1])
    if upload.error:
        print(f"Upload error: {upload.error}")
        sys.exit(1)

    doc = extract_text(upload)
    if doc.error:
        print(f"Extraction error: {doc.error}")
        sys.exit(1)

    doc = detect_layout(doc)
    doc = parse_sections(doc, llm_api_key=api_key)

    print(f"Sections found: {len(doc.sections)}")
    print("─" * 50)
    for sec in doc.sections:
        preview = sec.raw_text[:60].replace("\n", " | ")
        print(f"  [{sec.section_type.name:15s}] {sec.heading}")
        print(f"       {preview}…" if len(sec.raw_text) > 60 else f"       {preview}")
        print()

    if doc.parsed_resume:
        print("═" * 50)
        print("Structured output:")
        print(json.dumps(asdict(doc.parsed_resume), indent=2, default=str))
