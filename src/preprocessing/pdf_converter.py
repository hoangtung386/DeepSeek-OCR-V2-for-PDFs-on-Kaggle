"""Convert PDF files to PIL images using pdf2image."""

import logging
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


def convert_pdf_to_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    """Convert a PDF file into a list of PIL Images (one per page).

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering pages. Higher means better quality.

    Returns:
        List of PIL Images representing each page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info("Converting %s to images (DPI=%d)", path.name, dpi)
    images = convert_from_path(str(path), dpi=dpi)
    logger.info("Converted %s: %d page(s)", path.name, len(images))
    return images
