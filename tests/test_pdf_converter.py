"""Tests for PDF to image conversion."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Mock pdf2image before importing the module under test
sys.modules.setdefault("pdf2image", MagicMock())

from src.preprocessing.pdf_converter import convert_pdf_to_images  # noqa: E402


class TestConvertPdfToImages:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            convert_pdf_to_images("/nonexistent/file.pdf")

    @patch("src.preprocessing.pdf_converter.convert_from_path")
    def test_successful_conversion(self, mock_convert, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_images = [Image.new("RGB", (100, 100)) for _ in range(3)]
        mock_convert.return_value = mock_images

        result = convert_pdf_to_images(str(pdf_file), dpi=150)

        assert len(result) == 3
        mock_convert.assert_called_once_with(str(pdf_file), dpi=150)

    @patch("src.preprocessing.pdf_converter.convert_from_path")
    def test_default_dpi(self, mock_convert, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")
        mock_convert.return_value = []

        convert_pdf_to_images(str(pdf_file))
        mock_convert.assert_called_once_with(str(pdf_file), dpi=300)
