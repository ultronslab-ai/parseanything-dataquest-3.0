"""Scanned Document Fallback Pipeline Orchestrator.
Additive, non-destructive coordinator for scanned, handwritten, and degraded pages.
Feature-flagged via ENABLE_SCANNED_FALLBACK (default True).
Automatically falls back to baseline parser on any unexpected exception.
"""

import os
import traceback
from typing import Any, Dict, List, Optional
from PIL import Image

from schemas.models import SemanticBlock
from extractors.ocr.ocr_provider import get_ocr_provider
from .page_classifier import PageQualityClassifier, PageType, PageClassificationResult
from .image_preprocessor import ScannedImagePreprocessor, PreprocessedImageResult
from .region_detector import SemanticRegionDetector
from .block_reconstructor import SemanticBlockReconstructor

# Feature flag: can be toggled via environment variable without code changes
ENABLE_SCANNED_FALLBACK = os.getenv("ENABLE_SCANNED_FALLBACK", "true").lower() in ("true", "1", "yes")


class PageProcessingResult:
    """Encapsulates output of page-level scanned processing."""

    def __init__(
        self,
        success: bool,
        page_type: str,
        blocks: List[SemanticBlock],
        quality_score: float = 1.0,
        preprocessing_metadata: Optional[Dict[str, Any]] = None,
        is_digital: bool = False,
        error: Optional[str] = None
    ):
        self.success = success
        self.page_type = page_type
        self.blocks = blocks
        self.quality_score = quality_score
        self.preprocessing_metadata = preprocessing_metadata or {}
        self.is_digital = is_digital
        self.error = error


class ScannedDocumentPipeline:
    """Master orchestrator for the additive scanned/handwritten pipeline."""

    def __init__(self):
        self.ocr_provider = None
        self.reconstructor = SemanticBlockReconstructor()

    def _get_ocr(self):
        if self.ocr_provider is None:
            self.ocr_provider = get_ocr_provider()
        return self.ocr_provider

    def process_page(
        self,
        image: Image.Image,
        page_num: int,
        filename: str,
        page_w: float,
        page_h: float,
        native_text: str = "",
        embedded_images_count: int = 0,
        options: Optional[Dict[str, Any]] = None
    ) -> PageProcessingResult:
        """Executes the complete scanned page fallback pipeline with zero-crash guarantee."""
        if not ENABLE_SCANNED_FALLBACK:
            return PageProcessingResult(
                success=False,
                page_type="DISABLED",
                blocks=[],
                error="ENABLE_SCANNED_FALLBACK is disabled."
            )

        try:
            # 1. Page Quality & Type Classification
            classification: PageClassificationResult = PageQualityClassifier.classify(
                image=image,
                native_text=native_text,
                embedded_images_count=embedded_images_count,
                page_num=page_num
            )

            # If page is purely digital text, signal caller to use native vector path
            if classification.page_type == PageType.DIGITAL_TEXT:
                return PageProcessingResult(
                    success=True,
                    page_type=classification.page_type.value,
                    blocks=[],
                    quality_score=classification.quality_score,
                    is_digital=True
                )

            is_handwritten = classification.page_type in (PageType.HANDWRITTEN, PageType.MIXED)

            # 2. Image Preprocessing (Deskew, Illumination Normalization, CLAHE, Noise Filter)
            preprocessed: PreprocessedImageResult = ScannedImagePreprocessor.preprocess(
                image=image,
                is_handwritten=is_handwritten,
                quality_score=classification.quality_score
            )

            # 3. Base OCR Token Extraction on Enhanced Image
            ocr = self._get_ocr()
            ocr_tokens = ocr.extract_text(preprocessed.enhanced_image)

            # 4. Region & Block Detection (Anti-fragmentation grouping)
            regions = SemanticRegionDetector.group_tokens_into_regions(
                ocr_lines=ocr_tokens,
                page_width=float(preprocessed.width),
                page_height=float(preprocessed.height),
                page_num=page_num,
                page_type=classification.page_type.value
            )

            # 5. Semantic Block Reconstruction & Specialized Routing
            blocks = self.reconstructor.reconstruct_page_blocks(
                regions=regions,
                page_image=preprocessed.enhanced_image,
                page_num=page_num,
                filename=filename,
                page_w=page_w,
                page_h=page_h,
                is_handwritten_page=is_handwritten,
                quality_score=classification.quality_score
            )

            return PageProcessingResult(
                success=True,
                page_type=classification.page_type.value,
                blocks=blocks,
                quality_score=classification.quality_score,
                preprocessing_metadata=preprocessed.metadata,
                is_digital=False
            )

        except Exception as e:
            # Failure handling: log details and return failure for graceful fallback
            err_msg = f"ScannedDocumentPipeline encountered exception on page {page_num}: {str(e)}"
            print(f"[ScannedDocumentPipeline] WARNING: {err_msg}")
            traceback.print_exc()
            return PageProcessingResult(
                success=False,
                page_type="ERROR",
                blocks=[],
                error=err_msg
            )
