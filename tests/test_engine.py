"""Tests for the DeepSeek OCR engine (mocked, no GPU required)."""

import sys
from unittest.mock import MagicMock

# Mock heavy dependencies before importing the module under test
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("transformers", MagicMock())

from src.config import ModelConfig  # noqa: E402
from src.ocr.engine import DeepSeekOCREngine  # noqa: E402


class TestDeepSeekOCREngine:
    def test_initialization(self):
        config = ModelConfig(model_id="test/model")
        engine = DeepSeekOCREngine(config)

        assert engine.tokenizer is not None
        assert engine.model is not None

    def test_run_ocr_markdown(self):
        config = ModelConfig()
        engine = DeepSeekOCREngine(config)
        engine.model.infer.return_value = "# Extracted Markdown"

        result = engine.run_ocr("/tmp/test.png", task_type="markdown")

        assert result == "# Extracted Markdown"
        engine.model.infer.assert_called_once()

    def test_run_ocr_free_text(self):
        config = ModelConfig()
        engine = DeepSeekOCREngine(config)
        engine.model.infer.return_value = "Plain text output"

        result = engine.run_ocr("/tmp/test.png", task_type="ocr")

        assert result == "Plain text output"
        call_kwargs = engine.model.infer.call_args
        prompt = call_kwargs.kwargs.get("prompt") or call_kwargs[1].get("prompt", "")
        assert "Free OCR" in prompt

    def test_run_ocr_unknown_task_defaults_to_markdown(self):
        config = ModelConfig()
        engine = DeepSeekOCREngine(config)
        engine.model.infer.return_value = "markdown output"

        engine.run_ocr("/tmp/test.png", task_type="unknown")

        call_kwargs = engine.model.infer.call_args
        prompt = call_kwargs.kwargs.get("prompt") or call_kwargs[1].get("prompt", "")
        assert "Convert the document to markdown" in prompt
