"""Visualize docling-detected bounding boxes for each OCR engine."""

from pathlib import Path

# Reduce OCR rendering from 216 DPI (scale=3) to 144 DPI (scale=2) to stay
# within available memory. The detection model's preprocessing at scale=3
# creates >100 MB float64 intermediates on large pages, causing OOM crashes.
import docling.models.stages.ocr.rapid_ocr_model as _rapid_mod

_orig_rapid_init = _rapid_mod.RapidOcrModel.__init__


def _patched_rapid_init(self, *args, **kwargs):
    _orig_rapid_init(self, *args, **kwargs)
    self.scale = 2


_rapid_mod.RapidOcrModel.__init__ = _patched_rapid_init

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.visualization import draw_clusters

from compare_ocr import (
    MAX_PAGES,
    INPUT_PDF,
    _get_ppocr_v5_model_paths,
)

BOUNDINGS_DIR = INPUT_PDF.parent / "boundings"
IMAGES_SCALE = 2.0


def _make_converter(ocr_options: RapidOcrOptions) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = ocr_options
    pipeline_options.generate_page_images = True
    pipeline_options.images_scale = IMAGES_SCALE
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _save_page_visualizations(
    converter: DocumentConverter,
    pdf_path: Path,
    output_dir: Path,
    engine_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = converter.convert(str(pdf_path), page_range=(1, MAX_PAGES))

    saved = 0
    for page in result.pages:
        clusters = (
            page.predictions.layout.clusters
            if page.predictions.layout is not None
            else []
        )

        doc_page = result.document.pages.get(page.page_no)
        if doc_page is None or doc_page.image is None:
            continue
        pil_img = doc_page.image.pil_image
        if pil_img is None:
            continue

        vis = pil_img.copy()
        if clusters:
            draw_clusters(vis, clusters, scale_x=IMAGES_SCALE, scale_y=IMAGES_SCALE)

        out_path = output_dir / f"page_{page.page_no:04d}.png"
        vis.save(str(out_path))
        saved += 1

    print(f"  {engine_name}: saved {saved} pages → {output_dir}")


def visualize_rapidocr(pdf_path: Path) -> None:
    print("=== RapidOCR bounding boxes ===")
    ocr_options = RapidOcrOptions(
        force_full_page_ocr=True,
        use_cls=False,
        rapidocr_params={"Det.limit_type": "max"},
    )
    converter = _make_converter(ocr_options)
    _save_page_visualizations(
        converter, pdf_path, BOUNDINGS_DIR / "RapidOCR", "RapidOCR"
    )


def visualize_ppocr_v5(pdf_path: Path) -> None:
    print("=== PP-OCRv5 bounding boxes ===")
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
    _save_page_visualizations(
        converter, pdf_path, BOUNDINGS_DIR / "PP-OCRv5", "PP-OCRv5"
    )


if __name__ == "__main__":
    if not INPUT_PDF.exists():
        raise FileNotFoundError(INPUT_PDF)

    visualize_rapidocr(INPUT_PDF)
    visualize_ppocr_v5(INPUT_PDF)

    print("\nVisualization complete.")
