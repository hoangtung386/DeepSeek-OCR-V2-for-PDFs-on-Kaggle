# DeepSeek OCR V2 Pipeline for PDFs

Automated pipeline to extract text from PDF files using the **DeepSeek-OCR-2** vision-language model. Converts entire PDFs into clean Markdown output — preserving document structure, tables, and text layout.

Built to run on **Kaggle's free GPU resources** (T4 x2), but works on any NVIDIA GPU locally.

## How It Works

```
PDF files ─> pdf2image (page splitting) ─> DeepSeek-OCR-2 (per-page inference) ─> Markdown output
```

1. Scans `data/` for all `.pdf` / `.PDF` files
2. Converts each page to a high-resolution image (default 300 DPI)
3. Feeds each image to DeepSeek-OCR-2 with a markdown extraction prompt
4. Merges all pages and writes one `.md` file per PDF into `output/`

## Repository Structure

```
├── configs/
│   └── default.yaml                # All settings in one place (model, DPI, paths)
├── src/
│   ├── __init__.py
│   ├── __main__.py                 # Enables `python -m src`
│   ├── config.py                   # Dataclass-based config + YAML loader
│   ├── cli.py                      # CLI entrypoint (argparse)
│   ├── pipeline.py                 # Main orchestrator
│   ├── ocr/
│   │   └── engine.py               # DeepSeek model loading and inference
│   ├── preprocessing/
│   │   └── pdf_converter.py        # PDF to PIL Image conversion (pdf2image)
│   └── postprocessing/
│       └── markdown_writer.py      # Combine pages and write Markdown
├── tests/                          # 22 unit + integration tests
│   ├── test_config.py
│   ├── test_engine.py
│   ├── test_pdf_converter.py
│   ├── test_markdown_writer.py
│   └── test_pipeline.py
├── notebooks/
│   └── kaggle_example.ipynb        # Ready-to-run Kaggle notebook (auto-downloads data)
├── pyproject.toml                  # Project metadata + pinned dependencies
├── Makefile                        # Shortcut commands
└── README.md
```

## Quick Start

### Option 1: On Kaggle (free GPU)

1. Create a new Notebook on Kaggle — select **GPU T4 x2** accelerator.
2. Copy the contents of `notebooks/kaggle_example.ipynb` into the Kaggle notebook.
3. Run all cells — the notebook will:
   - Clone this repo from GitHub
   - Download PDF data automatically from Google Drive
   - Install dependencies and run the OCR pipeline
4. Use **Save Version → Save & Run All** to persist outputs as a Kaggle artifact.

### Option 2: Run Locally

**Prerequisites** (Ubuntu/Debian):

```bash
sudo apt-get update && sudo apt-get install -y poppler-utils
```

> `poppler-utils` provides `pdftoppm`, required by `pdf2image` to render PDF pages.

**Install**:

```bash
pip install -e ".[dev]"
```

**Run**:

```bash
# Place PDFs in data/, then:
python -m src --data-dir data --output-dir output

# With custom DPI and verbose logging:
python -m src --dpi 200 --verbose

# Or use the Makefile shortcut:
make run
```

**Run via Python** (e.g., in a Jupyter notebook):

```python
from src.pipeline import run_pipeline

run_pipeline(data_dir="data", output_dir="output")
```

## Configuration

All settings live in `configs/default.yaml`:

```yaml
model:
  model_id: "deepseek-ai/DeepSeek-OCR-2"
  torch_dtype: "float16"       # Use float16 for T4 GPU compatibility
  base_size: 1024
  image_size: 768
  crop_mode: true              # Efficient high-resolution processing

pdf:
  dpi: 300                     # Higher = better quality, larger images

paths:
  data_dir: "data"
  output_dir: "output"
```

CLI flags override YAML values:

| Flag | Description |
|------|-------------|
| `--config PATH` | Use a custom YAML config file |
| `--data-dir DIR` | Override input directory |
| `--output-dir DIR` | Override output directory |
| `--dpi N` | Override PDF rendering DPI |
| `--verbose` / `-v` | Enable debug-level logging |

## Development

```bash
make install    # Install project + dev dependencies (pytest, ruff)
make test       # Run 22 tests
make lint       # Check code style (ruff)
make lint-fix   # Auto-fix lint issues
make clean      # Remove __pycache__, build artifacts
```

### Running Tests

Tests mock `torch`, `transformers`, and `pdf2image` — no GPU needed:

```bash
make test
```

```
tests/test_config.py           6 passed
tests/test_engine.py           4 passed
tests/test_pdf_converter.py    3 passed
tests/test_markdown_writer.py  4 passed
tests/test_pipeline.py         5 passed
─────────────────────────────────────────
                              22 passed
```

## Technical Notes

- **GPU memory**: The model loads in `float16` with `device_map="auto"`, automatically distributing across available GPUs. Compatible with Kaggle's T4 (16 GB VRAM each).
- **Crop mode**: `crop_mode=True` splits large pages into tiles for the vision encoder, improving accuracy on dense documents.
- **Case-insensitive**: The pipeline picks up both `.pdf` and `.PDF` files on case-sensitive filesystems (Linux).
- **Safe temp files**: Page images use `tempfile.NamedTemporaryFile` — no hardcoded paths, safe for parallel execution.
