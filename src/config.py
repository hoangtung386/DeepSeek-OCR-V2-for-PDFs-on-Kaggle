"""Centralized configuration management using dataclasses and YAML."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Configuration for the DeepSeek OCR model."""

    model_id: str = "deepseek-ai/DeepSeek-OCR-2"
    torch_dtype: str = "float32"
    base_size: int = 1024
    image_size: int = 768
    crop_mode: bool = True


@dataclass
class PdfConfig:
    """Configuration for PDF processing."""

    dpi: int = 300


@dataclass
class PathsConfig:
    """Configuration for input/output paths."""

    data_dir: str = "data"
    output_dir: str = "output"


@dataclass
class AppConfig:
    """Top-level application configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    pdf: PdfConfig = field(default_factory=PdfConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "AppConfig":
        """Load configuration from a YAML file."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()
        if "model" in raw:
            config.model = ModelConfig(**raw["model"])
        if "pdf" in raw:
            config.pdf = PdfConfig(**raw["pdf"])
        if "paths" in raw:
            config.paths = PathsConfig(**raw["paths"])
        return config
