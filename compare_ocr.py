"""OCR engine comparison: RapidOCR (default models) vs PP-OCRv5 via RapidOCR."""

import time
import urllib.request
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_HERE = Path(__file__).parent
INPUT_PDF = _HERE / "examples/shido/input.pdf"
OUTPUT_DIR = _HERE / "examples/shido"
MODELS_CACHE = _HERE / ".models"
MAX_PAGES = 30  # process only the first N pages for faster comparison


def _make_converter(ocr_options: RapidOcrOptions) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _convert_and_save(
    converter: DocumentConverter, pdf_path: Path, output_path: Path
) -> float:
    start = time.perf_counter()
    result = converter.convert(str(pdf_path), page_range=(1, MAX_PAGES))
    elapsed = time.perf_counter() - start
    output_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    return elapsed


# ---------------------------------------------------------------------------
# PP-OCRv5 model download
# ---------------------------------------------------------------------------

_PPOCR_V5_HF_REPO = "RapidAI/PP-OCRv5"

_PPOCR_V5_HF_FILES = {
    "det": "det/ch_PP-OCRv5_det_mobile.onnx",
    "rec": "rec/ch_PP-OCRv5_rec_mobile.onnx",
    "cls": "cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
    "dict": "dict/ppocrv5_dict.txt",
}

# Fallback: ModelScope CDN (v3.8.0 tag)
_PPOCR_V5_MODELSCOPE_BASE = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0"
)
_PPOCR_V5_MODELSCOPE_FILES = {
    "det": f"{_PPOCR_V5_MODELSCOPE_BASE}/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
    "rec": f"{_PPOCR_V5_MODELSCOPE_BASE}/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx",
    "cls": f"{_PPOCR_V5_MODELSCOPE_BASE}/onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
    "dict": f"{_PPOCR_V5_MODELSCOPE_BASE}/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt",
}


def _download_ppocr_v5_via_hf() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths: dict[str, Path] = {}
    for key, filename in _PPOCR_V5_HF_FILES.items():
        paths[key] = Path(
            hf_hub_download(
                repo_id=_PPOCR_V5_HF_REPO,
                filename=filename,
                cache_dir=str(MODELS_CACHE),
            )
        )
    return paths


def _download_ppocr_v5_via_direct() -> dict[str, Path]:
    dest_dir = MODELS_CACHE / "ppocr_v5"
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for key, url in _PPOCR_V5_MODELSCOPE_FILES.items():
        filename = url.split("/")[-1]
        dest = dest_dir / filename
        if not dest.exists():
            print(f"  Downloading {filename} ...")
            urllib.request.urlretrieve(url, dest)
        paths[key] = dest
    return paths


def _get_ppocr_v5_model_paths() -> dict[str, Path]:
    """Download PP-OCRv5 ONNX models and return local paths.

    Tries HuggingFace first; falls back to ModelScope CDN.
    """
    MODELS_CACHE.mkdir(exist_ok=True)
    try:
        print("  Fetching PP-OCRv5 models from HuggingFace ...")
        return _download_ppocr_v5_via_hf()
    except Exception as e:
        print(f"  HuggingFace unavailable ({e}), trying ModelScope CDN ...")
        return _download_ppocr_v5_via_direct()


# ---------------------------------------------------------------------------
# OCR runs
# ---------------------------------------------------------------------------


def run_rapidocr(pdf_path: Path, output_path: Path) -> None:
    """Convert with RapidOCR using its built-in default models."""
    print("=== RapidOCR (default models) ===")
    ocr_options = RapidOcrOptions(
        force_full_page_ocr=True,
        use_cls=False,
        rapidocr_params={"Det.limit_type": "max"},
    )
    converter = _make_converter(ocr_options)
    elapsed = _convert_and_save(converter, pdf_path, output_path)
    print(f"  Completed in {elapsed:.1f}s  →  {output_path}\n")


def run_ppocr_v5(pdf_path: Path, output_path: Path) -> None:
    """Convert with PP-OCRv5 ONNX models via RapidOCR.

    PP-OCRv5's Chinese recognition model natively supports Japanese
    (Hiragana, Katakana, Kanji).
    """
    print("=== PP-OCRv5 (via RapidOCR) ===")
    models = _get_ppocr_v5_model_paths()
    ocr_options = RapidOcrOptions(
        force_full_page_ocr=True,
        use_cls=False,
        rapidocr_params={"Det.limit_type": "max"},
        det_model_path=str(models["det"]),
        rec_model_path=str(models["rec"]),
        cls_model_path=str(models["cls"]),
        rec_keys_path=str(models["dict"]),
    )
    converter = _make_converter(ocr_options)
    elapsed = _convert_and_save(converter, pdf_path, output_path)
    print(f"  Completed in {elapsed:.1f}s  →  {output_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not INPUT_PDF.exists():
        raise FileNotFoundError(INPUT_PDF)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_rapidocr(INPUT_PDF, OUTPUT_DIR / "output_RapidOCR.md")
    run_ppocr_v5(INPUT_PDF, OUTPUT_DIR / "output_PP-OCRv5.md")

    print("Comparison complete.")
