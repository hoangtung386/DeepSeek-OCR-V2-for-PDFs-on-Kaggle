"""Main pipeline orchestrating PDF-to-Markdown conversion."""

import logging
import tempfile
from pathlib import Path

from src.config import AppConfig
from src.ocr.engine import DeepSeekOCREngine
from src.postprocessing.markdown_writer import write_markdown
from src.preprocessing.pdf_converter import convert_pdf_to_images

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrate end-to-end PDF OCR processing."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.engine = DeepSeekOCREngine(config.model)
        Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)

    def process_pdf(self, pdf_path: Path) -> None:
        """Process a single PDF: convert pages to images, run OCR, save markdown."""
        output_file = Path(self.config.paths.output_dir) / f"{pdf_path.stem}.md"
        logger.info("Processing %s", pdf_path.name)

        images = convert_pdf_to_images(str(pdf_path), dpi=self.config.pdf.dpi)
        pages_text: list[str] = []

        for i, image in enumerate(images):
            logger.info("  Page %d/%d", i + 1, len(images))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                image.save(tmp.name)
                text = self.engine.run_ocr(tmp.name, task_type="markdown")
                pages_text.append(text)

        write_markdown(pages_text, str(output_file))

    def run(self) -> None:
        """Discover and process all PDFs in the data directory."""
        data_dir = Path(self.config.paths.data_dir)
        pdf_files = sorted(
            p for p in data_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        )

        if not pdf_files:
            logger.warning("No PDF files found in %s", data_dir)
            return

        logger.info("Found %d PDF(s) in %s", len(pdf_files), data_dir)
        for pdf in pdf_files:
            self.process_pdf(pdf)
        logger.info("All done!")


def run_pipeline(
    data_dir: str = "data",
    output_dir: str = "output",
    config_path: str | None = None,
) -> None:
    """Convenience function for Kaggle notebooks and quick usage."""
    if config_path:
        config = AppConfig.from_yaml(config_path)
    else:
        config = AppConfig()

    config.paths.data_dir = data_dir
    config.paths.output_dir = output_dir

    pipeline = Pipeline(config)
    pipeline.run()
