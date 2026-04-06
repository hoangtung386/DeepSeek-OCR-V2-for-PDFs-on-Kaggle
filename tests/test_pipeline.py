"""Integration tests for the pipeline."""

import sys
from unittest.mock import MagicMock, patch

from PIL import Image

# Mock heavy dependencies before importing
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("pdf2image", MagicMock())

from src.config import AppConfig, ModelConfig, PathsConfig, PdfConfig  # noqa: E402
from src.pipeline import Pipeline, run_pipeline  # noqa: E402


class TestPipeline:
    @patch("src.pipeline.DeepSeekOCREngine")
    @patch("src.pipeline.convert_pdf_to_images")
    def test_process_pdf(self, mock_convert, mock_engine_cls, tmp_path):
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        data_dir.mkdir()
        output_dir.mkdir()

        # Create a fake PDF file
        pdf_file = data_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        # Mock image conversion
        mock_images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
        mock_convert.return_value = mock_images

        # Mock OCR engine
        mock_engine = MagicMock()
        mock_engine.run_ocr.side_effect = ["# Page 1 content", "# Page 2 content"]
        mock_engine_cls.return_value = mock_engine

        config = AppConfig(
            model=ModelConfig(),
            pdf=PdfConfig(dpi=150),
            paths=PathsConfig(data_dir=str(data_dir), output_dir=str(output_dir)),
        )
        pipeline = Pipeline(config)
        pipeline.process_pdf(pdf_file)

        output_file = output_dir / "test.md"
        assert output_file.exists()

        content = output_file.read_text(encoding="utf-8")
        assert "Page 1 content" in content
        assert "Page 2 content" in content
        assert mock_engine.run_ocr.call_count == 2

    @patch("src.pipeline.DeepSeekOCREngine")
    def test_run_no_pdfs(self, mock_engine_cls, tmp_path):
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        data_dir.mkdir()
        output_dir.mkdir()

        config = AppConfig(
            paths=PathsConfig(data_dir=str(data_dir), output_dir=str(output_dir)),
        )
        pipeline = Pipeline(config)
        pipeline.run()  # Should not raise

    @patch("src.pipeline.DeepSeekOCREngine")
    @patch("src.pipeline.convert_pdf_to_images")
    def test_run_discovers_pdfs(self, mock_convert, mock_engine_cls, tmp_path):
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        data_dir.mkdir()
        output_dir.mkdir()

        # Create two fake PDFs
        (data_dir / "a.pdf").write_bytes(b"%PDF")
        (data_dir / "b.pdf").write_bytes(b"%PDF")

        mock_convert.return_value = [Image.new("RGB", (10, 10))]
        mock_engine = MagicMock()
        mock_engine.run_ocr.return_value = "text"
        mock_engine_cls.return_value = mock_engine

        config = AppConfig(
            paths=PathsConfig(data_dir=str(data_dir), output_dir=str(output_dir)),
        )
        Pipeline(config).run()

        assert (output_dir / "a.md").exists()
        assert (output_dir / "b.md").exists()


class TestRunPipeline:
    @patch("src.pipeline.Pipeline")
    def test_run_pipeline_default(self, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        run_pipeline(data_dir="/tmp/data", output_dir="/tmp/output")

        mock_pipeline.run.assert_called_once()

    @patch("src.pipeline.Pipeline")
    @patch("src.pipeline.AppConfig.from_yaml")
    def test_run_pipeline_with_config(self, mock_from_yaml, mock_pipeline_cls):
        mock_config = AppConfig()
        mock_from_yaml.return_value = mock_config

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        run_pipeline(
            data_dir="/tmp/data",
            output_dir="/tmp/output",
            config_path="custom.yaml",
        )

        mock_from_yaml.assert_called_once_with("custom.yaml")
        mock_pipeline.run.assert_called_once()
