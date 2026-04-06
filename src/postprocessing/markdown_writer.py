"""Write OCR results to Markdown files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_markdown(pages: list[str], output_path: str) -> None:
    """Combine page texts and write to a Markdown file.

    Args:
        pages: List of extracted text strings, one per page.
        output_path: Destination file path for the Markdown output.
    """
    sections = [
        f"<!-- Page {i + 1} -->\n{text}"
        for i, text in enumerate(pages)
    ]
    content = "\n\n---\n\n".join(sections)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    logger.info("Saved markdown to %s", output_path)
