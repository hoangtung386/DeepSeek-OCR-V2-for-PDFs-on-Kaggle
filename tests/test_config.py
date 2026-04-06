"""Tests for configuration loading and validation."""

import pytest

from src.config import AppConfig, ModelConfig, PathsConfig, PdfConfig


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_id == "deepseek-ai/DeepSeek-OCR-2"
        assert cfg.torch_dtype == "float16"
        assert cfg.base_size == 1024
        assert cfg.image_size == 768
        assert cfg.crop_mode is True

    def test_custom_values(self):
        cfg = ModelConfig(model_id="custom/model", torch_dtype="bfloat16", base_size=512)
        assert cfg.model_id == "custom/model"
        assert cfg.torch_dtype == "bfloat16"
        assert cfg.base_size == 512


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.model.model_id == "deepseek-ai/DeepSeek-OCR-2"
        assert cfg.pdf.dpi == 300
        assert cfg.paths.data_dir == "data"
        assert cfg.paths.output_dir == "output"

    def test_from_yaml(self, tmp_path):
        yaml_content = """
model:
  model_id: "test/model"
  torch_dtype: "bfloat16"
  base_size: 512
  image_size: 384
  crop_mode: false
pdf:
  dpi: 150
paths:
  data_dir: "/tmp/data"
  output_dir: "/tmp/output"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        cfg = AppConfig.from_yaml(str(config_file))
        assert cfg.model.model_id == "test/model"
        assert cfg.model.torch_dtype == "bfloat16"
        assert cfg.model.base_size == 512
        assert cfg.model.crop_mode is False
        assert cfg.pdf.dpi == 150
        assert cfg.paths.data_dir == "/tmp/data"

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AppConfig.from_yaml("/nonexistent/config.yaml")

    def test_from_yaml_partial(self, tmp_path):
        yaml_content = """
pdf:
  dpi: 200
"""
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(yaml_content)

        cfg = AppConfig.from_yaml(str(config_file))
        assert cfg.pdf.dpi == 200
        # Defaults preserved for unspecified sections
        assert cfg.model.model_id == "deepseek-ai/DeepSeek-OCR-2"
        assert cfg.paths.data_dir == "data"
