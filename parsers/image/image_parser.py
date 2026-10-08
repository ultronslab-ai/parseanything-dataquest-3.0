"""High-Fidelity Image Parser Adapter for ParseAnything.
Processes standalone raster images (PNG, JPG, JPEG) through image preprocessing,
local RapidOCR layout extraction, bounding box preservation, and confidence analysis.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from parsers.base import BaseParser
from schemas.models import (
    BlockStatus,
    BlockType,
    ConfidenceLevel,
    Document,
    DocumentMetadata,
    PageRenderInfo,
    ProcessingStats,
    SemanticBlock,
    SourceLocation,
)
from extractors.ocr.ocr_provider import get_ocr_provider
from extractors.equation.equation_extractor import EquationExtractor
from pipeline.reading_order.reading_order_engine import ReadingOrderEngine
from extractors.vision.region_detector import RegionDetector
from extractors.vision.math_ocr import MathOCRExtractor
from extractors.vision.handwriting_ocr import HandwritingOCRExtractor
from schemas.models import EquationData


class ImageParser(BaseParser):
    """Specialized parser for standalone images (PNG, JPG, JPEG)."""

    @property
    def supported_formats(self) -> List[str]:
        return ["png", "jpg", "jpeg"]

    def parse(
        self,
        file_path: Path,
        document_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        start_time = time.time()
        filename = file_path.name
        file_size = file_path.stat().st_size
        opts = options or {}
        images_dir = opts.get("images_output_dir", None)

        with Image.open(str(file_path)) as pil_img:
            img_w, img_h = pil_img.size
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Copy image to images_dir for frontend visualizer if requested
            rendered_image_url = None
            if images_dir:
                try:
                    out_path = Path(images_dir) / f"{document_id}_p1.png"
                    pil_img.save(str(out_path), format="PNG")
                    rendered_image_url = f"/api/v1/documents/{document_id}/page-image/1"
                except Exception as e:
                    pass

            ocr_provider = get_ocr_provider()
            ocr_lines = ocr_provider.extract_text(pil_img)
            
            math_ocr = MathOCRExtractor()
            hw_ocr = HandwritingOCRExtractor()

        # -------------------------------------------------------------
        # ADDITIVE SCANNED & HANDWRITTEN FALLBACK PIPELINE
        # -------------------------------------------------------------
        from scanned_pipeline.scanned_orchestrator import ScannedDocumentPipeline, ENABLE_SCANNED_FALLBACK
        image_processed = False
        page_type_classified = "SCANNED_PRINTED"
        page_quality = 1.0
        page_preproc_meta = {}

        all_blocks: List[SemanticBlock] = []
        warnings: List[str] = []
        equations_count = 0
        low_conf_count = 0

        if ENABLE_SCANNED_FALLBACK:
            try:
                pipeline = ScannedDocumentPipeline()
                res = pipeline.process_page(
                    image=pil_img,
                    page_num=1,
                    filename=filename,
                    page_w=float(img_w),
                    page_h=float(img_h),
                    options=opts
                )
                if res.success and res.blocks:
                    all_blocks = res.blocks
                    image_processed = True
                    page_type_classified = res.page_type
                    page_quality = res.quality_score
                    page_preproc_meta = res.preprocessing_metadata
                    equations_count = sum(1 for b in all_blocks if b.type == BlockType.EQUATION)
                    low_conf_count = sum(1 for b in all_blocks if b.confidence < 0.70)
            except Exception as e:
                warnings.append(f"Scanned fallback error on image: {e}. Falling back to baseline OCR.")
                image_processed = False

        if not image_processed:
            for line_idx, line in enumerate(ocr_lines):
                # Ensure bbox is valid before crop
                if line.bbox[2] <= line.bbox[0] or line.bbox[3] <= line.bbox[1]:
                    continue
                
                crop = pil_img.crop((line.bbox[0], line.bbox[1], line.bbox[2], line.bbox[3]))
                b_type = BlockType.PARAGRAPH
                eq_data = None
                final_text = line.text
                final_conf = round(line.confidence, 4)
                method = "ocr_rapid"
                
                if EquationExtractor.is_equation(line.text) or any(c in line.text for c in ['=', '+', '-', '√', '∫', '∑']):
                    latex_text = math_ocr.extract(crop)
                    if latex_text:
                        b_type = BlockType.EQUATION
                        final_text = latex_text
                        eq_data = EquationData(latex=latex_text, is_inline=False)
                        method = "math_ocr"
                        final_conf = 0.92
                        equations_count += 1
                    else:
                        eq_data, eq_conf = EquationExtractor.extract_equation(line.text)
                        b_type = BlockType.EQUATION
                        
                elif line.confidence < 0.85:
                    hw_text = hw_ocr.extract(crop)
                    if hw_text:
                        final_text = hw_text
                        method = "handwriting_ocr"
                        final_conf = max(line.confidence, 0.85)
                        
                elif line_idx == 0 and len(line.text.split()) <= 8:
                    b_type = BlockType.HEADING

                all_blocks.append(
                    SemanticBlock(
                        block_id=f"img_b_{line_idx+1:03d}",
                        type=b_type,
                        content=final_text,
                        source=SourceLocation(
                            file=filename,
                            page=1,
                            bbox=line.bbox,
                            coordinate_system="pixel"
                        ),
                        confidence=final_conf,
                        confidence_level=ConfidenceLevel.HIGH if final_conf >= 0.90 else (ConfidenceLevel.MEDIUM if final_conf >= 0.70 else ConfidenceLevel.LOW),
                        extraction_method=method,
                        status=BlockStatus.ACCEPTED if final_conf >= 0.70 else BlockStatus.REVIEW_REQUIRED,
                        equation_data=eq_data
                    )
                )

        # Reconstruct reading order across columns
        ordered_blocks = ReadingOrderEngine.reconstruct_reading_order(
            all_blocks,
            page_width=float(img_w),
            page_height=float(img_h)
        )

        elapsed = time.time() - start_time

        return Document(
            document_id=document_id,
            filename=filename,
            file_type="image",
            file_size_bytes=file_size,
            processing_status="success",
            metadata=DocumentMetadata(
                title=filename,
                page_count=1
            ),
            blocks=ordered_blocks,
            pages=[
                PageRenderInfo(
                    page_number=1,
                    width=float(img_w),
                    height=float(img_h),
                    image_url=rendered_image_url,
                    page_type=page_type_classified,
                    quality_score=page_quality,
                    preprocessing_metadata=page_preproc_meta
                )
            ],
            warnings=warnings,
            errors=[],
            stats=ProcessingStats(
                processing_time_seconds=round(elapsed, 3),
                pages_processed=1,
                pages_per_second=round(1.0 / max(0.001, elapsed), 2),
                blocks_extracted=len(ordered_blocks),
                tables_detected=0,
                figures_detected=0,
                equations_detected=equations_count,
                low_confidence_blocks=low_conf_count
            )
        )
