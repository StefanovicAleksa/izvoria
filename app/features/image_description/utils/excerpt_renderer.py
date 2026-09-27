# Path: app/features/image_description/utils/excerpt_renderer.py

from app.core.language.domain.entities import ImageRules
from app.features.image_description.domain.entities import ImageDescription


def _sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].upper() + text[1:] + "." if text else ""


def render_excerpt(description: ImageDescription, rules: ImageRules) -> str:
    """
    The citable text of an image: the vision model's description, then the text on the image. Each block names
    who wrote it, because nothing here is the client's own wording:

        Image (machine description): A photographed whiteboard with a hand-drawn timeline. Marked: an arrow at "Q3".
        Text on the image (read by OCR):
        Q3 launch
        Budget review 14 March

    Without OCR the vision model's text items follow instead, labelled unverified.
    """
    blocks = []
    parts = [_sentence(description.shows)]
    if description.marks:
        parts.append(f"Marked: {description.marks.strip().rstrip('.')}.")
    described = " ".join(part for part in parts if part)
    if described:
        blocks.append(f"{rules.description_label}: {described}")

    if description.ocr_text.strip():
        blocks.append(f"{rules.ocr_text_label}:\n{description.ocr_text.strip()}")
    elif description.unverified_items:
        quoted = "; ".join(f'"{item}"' for item in description.unverified_items)
        blocks.append(f"{rules.unverified_text_label}: {quoted}")

    # Absence is a finding: the image holds text a question might be about, and the index doesn't. A withheld
    # legend glyph or stray dash ("■", "-") held no words, so it is no reason to say text is missing.
    if any(any(ch.isalnum() for ch in line) for line in description.ocr_withheld_lines):
        blocks.append(rules.withheld_text_note)

    return "\n".join(blocks)


def render_context(description: ImageDescription) -> str:
    """The chunk's retrieval context: file name, date taken, size. No words, so no language to get wrong."""
    parts = [description.file_name]
    if description.taken_at:
        parts.append(description.taken_at)
    if description.width and description.height:
        parts.append(f"{description.width}×{description.height}")
    return " · ".join(parts)
