"""Master Pipeline Orchestrator for ParseAnything.
Implements the authoritative 3-stage architecture:
Stage 1: Detect & Route
Stage 2: Extract & Assemble
Stage 3: Flag & Fail Safe
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from schemas.errors import ErrorCode, ProcessingError
from schemas.models import Document
from pipeline.detection.format_detector import FormatDetector
from pipeline.table_merge.table_merger import TableMerger
from pipeline.confidence.confidence_engine import ConfidenceEngine
from pipeline.fail_safe.fail_safe_engine import FailSafeEngine
from pipeline.assembly.markdown_generator import MarkdownGenerator

from parsers.pdf.pdf_parser import PDFParser
from parsers.docx.docx_parser import DOCXParser
from parsers.pptx.pptx_parser import PPTXParser
from parsers.xlsx.xlsx_parser import XLSXParser
from parsers.image.image_parser import ImageParser


class ExtractionPipeline:
    """Universal document ingestion orchestrator."""

    def __init__(self):
        self.parsers = {
            "pdf_parser": PDFParser(),
            "docx_parser": DOCXParser(),
            "pptx_parser": PPTXParser(),
            "xlsx_parser": XLSXParser(),
            "image_parser": ImageParser(),
        }

    def process(
        self,
        file_path: Path,
        document_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Executes the 3-stage pipeline end-to-end on any document."""
        start_time = time.time()
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        filename = file_path.name
        opts = options or {}

        # -------------------------------------------------------------
        # STAGE 1: DETECT & ROUTE
        # -------------------------------------------------------------
        detection = FormatDetector.detect(file_path)

        if not detection.is_supported or detection.error:
            elapsed = time.time() - start_time
            err = detection.error or ProcessingError(
                code=ErrorCode.UNSUPPORTED_FORMAT,
                message=f"Unsupported or unrecognized file format: {file_path.suffix}",
                stage="detect_and_route"
            )
            return FailSafeEngine.create_failure_document(
                document_id=doc_id,
                filename=filename,
                file_type=detection.format_name,
                error=err,
                processing_time=round(elapsed, 3)
            )

        parser = self.parsers.get(detection.parser_name)
        if not parser:
            elapsed = time.time() - start_time
            err = ProcessingError(
                code=ErrorCode.PARSER_FAILURE,
                message=f"No parser adapter registered for '{detection.parser_name}'",
                stage="detect_and_route"
            )
            return FailSafeEngine.create_failure_document(
                document_id=doc_id,
                filename=filename,
                file_type=detection.format_name,
                error=err,
                processing_time=round(elapsed, 3)
            )

        # -------------------------------------------------------------
        # STAGE 2: EXTRACT & ASSEMBLE
        # -------------------------------------------------------------
        try:
            document = parser.parse(file_path, doc_id, opts)
        except Exception as e:
            elapsed = time.time() - start_time
            err = ProcessingError(
                code=ErrorCode.PARSER_FAILURE,
                message=f"Parser execution failed on {filename}: {str(e)}",
                stage="extract_and_assemble",
                details={"exception_type": type(e).__name__}
            )
            return FailSafeEngine.create_failure_document(
                document_id=doc_id,
                filename=filename,
                file_type=detection.format_name,
                error=err,
                processing_time=round(elapsed, 3)
            )

        # Multi-page table merge across page boundaries
        document.blocks = TableMerger.merge_cross_page_tables(document.blocks)

        # -------------------------------------------------------------
        # STAGE 3: FLAG & FAIL SAFE
        # -------------------------------------------------------------
        # Multi-signal confidence calculation and review flagging
        document.blocks = ConfidenceEngine.evaluate_all(document.blocks)

        # Generate structured Markdown respecting reading order and hierarchy
        document.markdown = MarkdownGenerator.generate(document.blocks, filename)

        # Finalize timing and throughput metrics
        total_elapsed = round(time.time() - start_time, 3)
        document.stats.processing_time_seconds = total_elapsed
        if total_elapsed > 0:
            document.stats.pages_per_second = round(document.stats.pages_processed / total_elapsed, 2)
        document.stats.blocks_extracted = len(document.blocks)
        document.stats.low_confidence_blocks = sum(
            1 for b in document.blocks if b.confidence < ConfidenceEngine.MEDIUM_THRESHOLD
        )
        if document.blocks:
            document.stats.average_confidence = round(
                sum(b.confidence for b in document.blocks) / len(document.blocks), 4
            )

        return document
