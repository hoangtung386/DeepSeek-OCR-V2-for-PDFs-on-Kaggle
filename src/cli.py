"""Command-line interface for the DeepSeek OCR pipeline."""

import argparse
import logging

from src.config import AppConfig
from src.pipeline import Pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ocr-pipeline",
        description="Extract text from PDFs using DeepSeek OCR V2",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument("--data-dir", help="Override data input directory")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--dpi", type=int, help="Override PDF rendering DPI")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = AppConfig.from_yaml(args.config)

    if args.data_dir:
        config.paths.data_dir = args.data_dir
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.dpi:
        config.pdf.dpi = args.dpi

    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
