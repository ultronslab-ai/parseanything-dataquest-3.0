"""Master Document Extraction Pipeline.

Coordinates extraction engines, schema validation, visualization,
and multi-format export (JSON, CSV, annotated images).
"""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from doc_extractor.base import BaseExtractor
from doc_extractor.gemini_extractor import GeminiVisionExtractor
from doc_extractor.hybrid_extractor import HybridExtractor
from doc_extractor.models import DocumentExtractionResult
from doc_extractor.visualizer import DocumentVisualizer


class DocumentExtractionPipeline:
    """End-to-end pipeline for parsing PDFs and images into structured element streams."""

    def __init__(
        self,
        engine: Literal["hybrid", "gemini"] = "hybrid",
        gemini_api_key: Optional[str] = None,
        render_dpi: int = 150,
        enable_visualization: bool = True
    ):
        self.engine_type = engine
        self.render_dpi = render_dpi
        self.enable_visualization = enable_visualization
        self.visualizer = DocumentVisualizer() if enable_visualization else None

        if engine == "gemini":
            self.extractor: BaseExtractor = GeminiVisionExtractor(api_key=gemini_api_key)
        else:
            self.extractor: BaseExtractor = HybridExtractor(render_dpi=render_dpi)

    def process(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        export_json: bool = True
    ) -> DocumentExtractionResult:
        """Runs extraction on a document, saves outputs, and creates annotated visuals."""
        input_path = Path(file_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        # Run extraction engine
        result = self.extractor.extract_document(input_path)

        # Setup output destination
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Export Visual Annotations
            if self.enable_visualization and self.visualizer:
                annotated_paths = self.visualizer.render_and_annotate_pdf(
                    pdf_path=input_path,
                    page_results=result.pages,
                    output_dir=out_path / "annotated_pages",
                    dpi=self.render_dpi
                )
                for page_res, ann_path in zip(result.pages, annotated_paths):
                    page_res.annotated_image_path = ann_path

            # Export JSON
            if export_json:
                json_dest = out_path / f"{input_path.stem}_extracted.json"
                with open(json_dest, "w", encoding="utf-8") as f:
                    f.write(result.model_dump_json(indent=2))

        return result
