"""DeepSeek OCR V2 model loading and inference."""

import logging
import tempfile

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import ModelConfig

logger = logging.getLogger(__name__)

PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "ocr": "<image>\nFree OCR.",
}


def _patch_masked_scatter():
    """Patch masked_scatter_ to auto-cast source dtype to match target.

    DeepSeek-OCR-2's infer() hardcodes autocast to bfloat16, which T4 GPUs
    don't support. This causes the vision encoder to output float32 while
    text embeddings stay float16, crashing at masked_scatter_.
    """
    _orig = torch.Tensor.masked_scatter_

    def _safe_masked_scatter_(self, mask, source):
        if source.dtype != self.dtype:
            source = source.to(self.dtype)
        return _orig(self, mask, source)

    torch.Tensor.masked_scatter_ = _safe_masked_scatter_


class DeepSeekOCREngine:
    """Wrapper around DeepSeek-OCR-2 for document text extraction."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load tokenizer and model onto available GPU(s)."""
        logger.info("Loading tokenizer from %s", self.config.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )

        dtype = getattr(torch, self.config.torch_dtype)
        logger.info("Loading model from %s (dtype=%s)", self.config.model_id, dtype)
        # Force all weights onto a single GPU to avoid cross-device tensor errors.
        # DeepSeek-OCR-2's infer() code doesn't handle multi-GPU placement correctly.
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": 0},
        ).eval()
        _patch_masked_scatter()
        logger.info("Model loaded successfully")

    def run_ocr(self, image_path: str, task_type: str = "markdown") -> str:
        """Run OCR inference on a single image.

        Args:
            image_path: Path to the image file.
            task_type: "markdown" for structural extraction, "ocr" for free text.

        Returns:
            Extracted text as a string.
        """
        prompt = PROMPTS.get(task_type, PROMPTS["markdown"])

        with torch.no_grad():
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=tempfile.mkdtemp(),
                base_size=self.config.base_size,
                image_size=self.config.image_size,
                crop_mode=self.config.crop_mode,
            )
        return result
