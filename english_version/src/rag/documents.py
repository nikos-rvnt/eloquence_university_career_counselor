from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import config
import normalization

logger = logging.getLogger(__name__)


# the metadata fields that are used by the source files.
METADATA_PATTERNS = {
    "university": re.compile(r"^\s*UNIVERSITY\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
    "degree": re.compile(r"^\s*DEGREE\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
    "course": re.compile(r"^\s*COURSE\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
    "type": re.compile(r"^\s*TYPE\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
    "curriculum": re.compile(r"^\s*CURRICULUM\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
    "section": re.compile(r"^\s*SECTION\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
    "url": re.compile(r"^\s*URL\s*:\s*(https?://\S+)\s*$", re.IGNORECASE | re.MULTILINE),
    "region": re.compile(r"^\s*REGION\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
}

# a line of three or more hyphens may separate records.
RECORD_SEPARATOR_PATTERN = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)
# used as a fallback when there are no explicit separators.
UNIVERSITY_HEADER_PATTERN = re.compile(r"^\s*UNIVERSITY\s*:", re.IGNORECASE | re.MULTILINE)
URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def clean_text(text: str) -> str:
    """normalize line endings and remove common invisible characters."""
    return text.replace("\ufeff", "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _clean_value(value: str | None) -> str | None:
    """normalize whitespace and convert empty values to None."""
    if value is None:
        return None

    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _extract_field(text: str, field: str) -> str | None:
    """extract a metadata field from one record."""
    pattern = METADATA_PATTERNS.get(field)

    if pattern is None:
        raise ValueError(f"Unsupported metadata field: {field}")

    match = pattern.search(text)
    if not match:
        return None

    return _clean_value(match.group(1))


def parse_course_field(value: str | None) -> tuple[str | None, str | None]:
    """
    Split a COURSE value into its optional code and title.
    Examples:
        "[LM-9] BIOTECHNOLOGY" -> ("LM-9", "BIOTECHNOLOGY")
        "MEDICAL AND PHARMACEUTICAL BIOTECHNOLOGIES" -> (None, "MEDICAL AND PHARMACEUTICAL BIOTECHNOLOGIES")
    """
    value = _clean_value(value)
    if not value:
        return None, None

    match = re.match(r"^\[([^\]]+)\]\s*(.*?)\s*$", value)
    if match:
        return (_clean_value(match.group(1)), _clean_value(match.group(2)))

    return None, value


def load_documents() -> list[tuple[str, str]]:
    """
    Read all non-empty .txt files in config.DATA_DIR.
    Returns:
        [(absolute_source_path, full_text), ...]
    Source files are never modified.
    """
    data_dir = Path(config.DATA_DIR)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"University document directory does not exist: {data_dir}")

    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise RuntimeError(f"No .txt documents found in {data_dir}")

    loaded_docs = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            logger.error("Could not read %s: %s", path, exc)
            continue

        text = clean_text(text)
        if not text:
            logger.warning("Skipping empty file: %s", path)
            continue

        loaded_docs.append((str(path.resolve()), text))

    logger.info("Loaded %d source files.", len(loaded_docs))
    return loaded_docs


def split_document_records(source: str, text: str) -> list[tuple[str, str]]:
    """
    Split a source document into logical records.
    -> The RAG source files use explicit lines of hyphens (e.g. '-----')
    as the primary record boundary. These separators are therefore
    authoritative.
    If a source file does not contain explicit separators, repeated
    UNIVERSITY headers are used as a fallback boundary.
    -> COURSE fields are intentionally NOT used as boundaries because
    the source format can contain multiple section records belonging
    to the same university programme.
    Returns:
        List of (source, record_text) tuples.
    """

    text = clean_text(text)

    if not text:
        return []

    # Primary strategy: explicit record separators
    if RECORD_SEPARATOR_PATTERN.search(text):

        pieces = RECORD_SEPARATOR_PATTERN.split(text)
        records: list[tuple[str, str]] = []

        for piece in pieces:
            piece = clean_text(piece)

            if not piece:
                continue

            records.append((source, piece))

        logger.debug("Split %s into %d separator-based records.", Path(source).name, len(records))

        return records

    # Fallback strategy: repeated UNIVERSITY headers
    headers = list(UNIVERSITY_HEADER_PATTERN.finditer(text))

    if len(headers) <= 1:
        return [(source, text)]

    records: list[tuple[str, str]] = []

    # Preserve text before the first UNIVERSITY header.
    preamble = clean_text(text[: headers[0].start()])

    if preamble:
        records.append((source, preamble))

    for i, header in enumerate(headers):

        start = header.start()
        if i + 1 < len(headers):
            end = headers[i + 1].start()
        else:
            end = len(text)

        record = clean_text(text[start:end])
        if record:
            records.append((source, record))

    logger.debug("Split %s into %d UNIVERSITY-header-based records.", Path(source).name, len(records))

    return records


def _normalize_for_matching(value: str) -> str:
    """Normalize punctuation and whitespace for city-name matching."""
    value = value.casefold().replace("’", "'")
    value = re.sub(r"[_\-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _find_region_in_text(text: str) -> str | None:
    """Find a configured city name and return its normalized region."""
    normalized_text = _normalize_for_matching(text)

    # Match longer city names first.
    city_entries = sorted(normalization.CITY_TO_REGION.items(), key=lambda item: len(item[0]), reverse=True)

    for city, region in city_entries:
        city_normalized = _normalize_for_matching(city)

        if not city_normalized:
            continue

        pattern = r"(?<!\w)" + re.escape(city_normalized) + r"(?!\w)"

        if re.search(pattern, normalized_text):
            normalized_region = normalization.normalize_region(region)

            if normalized_region in config.VALID_REGIONS:
                return normalized_region

    return None


def infer_university_region(university_name: str | None, source_text: str, source: str | None = None) -> str | None:
    """
    Infer the university's geographic macro-region.
    Priority:
        1. Explicit REGION field in the current record.
        2. City name in the university name.
        3. City name in the source filename.
        4. City name in the current record text.
    This function does not use the student's home region.
    """
    explicit_region = _extract_field(source_text, "region")

    if explicit_region:
        normalized = normalization.normalize_region(explicit_region)

        if normalized in config.VALID_REGIONS:
            return normalized

    if university_name:
        region = _find_region_in_text(university_name)

        if region:
            return region

    if source:
        region = _find_region_in_text(Path(source).stem)

        if region:
            return region

    return _find_region_in_text(source_text)


def extract_record_metadata(source: str, record_text: str) -> dict[str, Any]:
    """
    Extract metadata from one record, before it is chunked.
    Do not pass the entire merged file here: pass one record returned
    by split_document_records().
    """
    record_text = clean_text(record_text)

    university = _extract_field(record_text, "university")
    raw_course = _extract_field(record_text, "course")
    course_code, course_title = parse_course_field(raw_course)

    course_type = _extract_field(record_text, "type")
    degree = _extract_field(record_text, "degree")
    curriculum = _extract_field(record_text, "curriculum")
    section = _extract_field(record_text, "section")
    url = _extract_field(record_text, "url")
    explicit_region = _extract_field(record_text, "region")

    if not url:
        url_match = URL_PATTERN.search(record_text)
        if url_match:
            url = url_match.group(0).rstrip(".,;)")
    else:
        url = url.rstrip(".,;)")

    region = infer_university_region(university_name=university, source_text=record_text, source=source)

    return {
        "university": university,
        "course": course_title,
        "course_code": course_code,
        "degree": degree,
        "type": course_type,
        "curriculum": curriculum,
        "section": section,
        "url": url,
        "region": region,
        "explicit_region": explicit_region,
    }


def format_metadata_prefix(metadata: dict[str, Any]) -> str:
    """
    Build a compact prefix for chunks.
    The URL is retained in metadata but omitted from the embedding text,
    since the URL itself usually contributes little semantic meaning.
    """
    fields = [
        ("UNIVERSITY", metadata.get("university")),
        ("COURSE", metadata.get("course")),
        ("DEGREE", metadata.get("degree")),
        ("COURSE_CODE", metadata.get("course_code")),
        ("TYPE", metadata.get("type")),
        ("CURRICULUM", metadata.get("curriculum")),
        ("SECTION", metadata.get("section")),
        ("REGION", metadata.get("region")),
    ]

    return "\n".join(f"{key}: {value}" for key, value in fields if value)


def extract_course_names_from_text(text: str) -> list[str]:
    """
    Extract course names from a merged document.
    Explicit COURSE fields take priority. Fallback patterns cover
    common degree-title formats when no COURSE field is available.
    CURRICULUM is deliberately not treated as a degree programme:
    a curriculum is usually a track within a programme.
    """

    # COURSE: [CODE] NAME / COURSE: NAME
    course_pattern = re.compile(
        r"COURSE\s*:\s*" r"(?:\[[^\]]+\]\s*)?" r"([^\n]+)",
        re.IGNORECASE,
    )
    #  Bachelor's / Bachelor Degree in NAME - Master's / Master Degree in NAME
    degree_pattern = re.compile(
        r"\b(?:Bachelor|Master)" r"(?:'s)?" r"\s+Degree" r"(?:\s+Program)?" r"\s+in\s+" r"([^\n,.]+)",
        re.IGNORECASE,
    )
    # LM-XX - NAME
    lm_pattern = re.compile(r"\bLM-\d+\s*[-–—]\s*([^\n]+)", re.IGNORECASE)
    # Degree Program in NAME
    program_pattern = re.compile(r"\bDegree\s+Program\s+in\s+" r"([^\n,.]+)", re.IGNORECASE)
    # CURRICULUM: NAME
    curriculum_pattern = re.compile(r"CURRICULUM\s*:\s*([^\n]+)", re.IGNORECASE)

    if not text or not text.strip():
        return []

    names = set()

    # use explicit COURSE fields record by record.
    records = split_document_records("<text>", text)

    for _, record in records:
        raw_course = _extract_field(record, "course")
        _, title = parse_course_field(raw_course)

        if title and len(title) > 3:
            names.add(title)

    # Fallbacks for records without explicit COURSE metadata.
    fallback_patterns = [
        re.compile(r"\b(?:Bachelor'?s|Bachelor)\s+Degree" r"(?:\s+Program)?\s+in\s+([^\n,.;]+)", re.IGNORECASE),
        re.compile(r"\b(?:Master'?s|Master)\s+Degree" r"(?:\s+Program)?\s+in\s+([^\n,.;]+)", re.IGNORECASE),
        re.compile(r"\bDegree\s+Program\s+in\s+([^\n,.;]+)", re.IGNORECASE),
        re.compile(r"\bLM-\d+\s*[-–—]\s*([^\n.;]+)", re.IGNORECASE),
    ]

    for _, record in records:
        # no fallback if the record contains the course name
        if _extract_field(record, "course"):
            continue

        for pattern in fallback_patterns:
            for match in pattern.findall(record):
                name = re.sub(r"\s+", " ", match).strip()
                name = name.strip(" \t\r\n:;,.|-–—")

                if len(name) > 3:
                    names.add(name)

    return sorted(names, key=str.casefold)
