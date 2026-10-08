"""Scanned and Handwritten Document Fallback Pipeline.
Additive, non-destructive pipeline for processing scanned, handwritten,
and degraded document pages with semantic region reconstruction.
"""

from .scanned_orchestrator import ScannedDocumentPipeline, ENABLE_SCANNED_FALLBACK
from .page_classifier import PageQualityClassifier, PageType
from .image_preprocessor import ScannedImagePreprocessor
from .region_detector import SemanticRegionDetector
from .handwriting_ocr import HandwritingOCRService
from .equation_ocr import EquationOCRService
from .confidence_recovery import ConfidenceRecoveryEngine
from .block_reconstructor import SemanticBlockReconstructor

__all__ = [
    "ScannedDocumentPipeline",
    "ENABLE_SCANNED_FALLBACK",
    "PageQualityClassifier",
    "PageType",
    "ScannedImagePreprocessor",
    "SemanticRegionDetector",
    "HandwritingOCRService",
    "EquationOCRService",
    "ConfidenceRecoveryEngine",
    "SemanticBlockReconstructor",
]
