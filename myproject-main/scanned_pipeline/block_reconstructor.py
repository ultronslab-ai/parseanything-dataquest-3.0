"""Semantic Block Reconstructor for Scanned & Handwritten Documents.
Converts layout regions into structured SemanticBlock instances with multi-column reading order,
honest confidence scoring, provenance coordinates, and full table/equation/chart payload support.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from schemas.models import (
    BlockStatus,
    BlockType,
    ConfidenceLevel,
    EquationData,
    SemanticBlock,
    SourceLocation,
    TableData,
    ChartData
)
from pipeline.reading_order.reading_order_engine import ReadingOrderEngine
from .region_detector import SemanticRegion, SemanticRegionDetector
from .handwriting_ocr import HandwritingOCRService
from .equation_ocr import EquationOCRService
from .confidence_recovery import ConfidenceRecoveryEngine
from .gemini_verifier import get_gemini_verifier


class SemanticBlockReconstructor:
    """Reconstructs semantic blocks from layout regions."""

    def __init__(self):
        self.hw_service = HandwritingOCRService()
        self.eq_service = EquationOCRService()
        self.recovery_engine = ConfidenceRecoveryEngine()
        self.verifier = get_gemini_verifier()
        self.review_threshold = self.verifier.review_threshold

    def reconstruct_page_blocks(
        self,
        regions: List[SemanticRegion],
        page_image: Image.Image,
        page_num: int,
        filename: str,
        page_w: float,
        page_h: float,
        is_handwritten_page: bool = False,
        quality_score: float = 1.0
    ) -> List[SemanticBlock]:
        """Converts semantic regions into final ordered SemanticBlock objects."""
        if not regions:
            return []

        img_w, img_h = page_image.size
        scale_x = page_w / float(img_w)
        scale_y = page_h / float(img_h)

        blocks: List[SemanticBlock] = []

        gemini_calls_count = 0
        max_gemini_calls = int(os.getenv("MAX_GEMINI_CALLS_PER_PAGE", "6"))

        for idx, reg in enumerate(regions):
            block_id = f"b_p{page_num}_{idx+1:03d}"
            reg_type = reg.region_type

            # Convert bbox to point coordinate system
            scaled_bbox = [
                round(reg.bbox[0] * scale_x, 2),
                round(reg.bbox[1] * scale_y, 2),
                round(reg.bbox[2] * scale_x, 2),
                round(reg.bbox[3] * scale_y, 2)
            ]

            # Crop region from page image for specialized routing
            crop_x0 = max(0, int(reg.bbox[0]))
            crop_y0 = max(0, int(reg.bbox[1]))
            crop_x1 = min(img_w, int(reg.bbox[2]))
            crop_y1 = min(img_h, int(reg.bbox[3]))

            crop = None
            if crop_x1 > crop_x0 and crop_y1 > crop_y0:
                crop = page_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))

            b_type = BlockType.PARAGRAPH
            final_content = reg.text
            final_conf = reg.confidence
            method = "ocr_local"
            model_name_val = "RapidOCR"
            requires_review = False
            alternatives: List[str] = []
            eq_data: Optional[EquationData] = None
            tbl_data: Optional[TableData] = None
            cht_data: Optional[ChartData] = None
            warnings: List[str] = []

            # =========================================================
            # ROUTE REGION ACCORDING TO SPECIFICATION
            # =========================================================

            # 1. EQUATION REGION
            if reg_type == "EQUATION" and crop is not None:
                eq_res = self.eq_service.process_equation(crop, reg.text, scaled_bbox, page_num)
                if eq_res.get("is_not_equation"):
                    # Gemini determined this was normal text/heading (e.g. law name), NOT an equation
                    b_type = BlockType.HEADING if len(eq_res["text"].split()) <= 6 else BlockType.PARAGRAPH
                    final_content = eq_res["text"]
                    eq_data = None
                    final_conf = eq_res.get("confidence", 0.85)
                    method = eq_res.get("method", "ocr_local+gemini_verification")
                    model_name_val = eq_res.get("model") or self.verifier.model_name
                    requires_review = eq_res.get("requires_review", False)
                    gemini_calls_count += 1
                else:
                    b_type = BlockType.EQUATION
                    final_content = eq_res["latex"]
                    eq_data = EquationData(latex=eq_res["latex"], is_inline=False)
                    final_conf = eq_res["confidence"]
                    method = eq_res.get("method", "ocr_local+gemini_verification")
                    model_name_val = eq_res.get("model") or self.verifier.model_name
                    requires_review = eq_res.get("requires_review", False)
                    if "gemini" in method:
                        gemini_calls_count += 1
                    if requires_review:
                        warnings.append("Equation structure requires verification.")

            # 2a. LINE_ITEM REGION (Financial Statement Row)
            elif reg_type == "LINE_ITEM":
                b_type = BlockType.LINE_ITEM
                method = "financial_statement_row"
                model_name_val = "financial_layout_analyzer"
                final_content = reg.text
                if final_conf < self.review_threshold and crop is not None:
                    # Low-confidence financial row: verify with Gemini if available, else recovery loop
                    gemini_handled = False
                    if self.verifier.is_available() and gemini_calls_count < max_gemini_calls:
                        gem_res = self.verifier.verify_and_enhance_block(
                            crop=crop,
                            ocr_text=reg.text,
                            bbox=scaled_bbox,
                            current_type="line_item",
                            current_conf=final_conf,
                            page_num=page_num,
                            is_handwritten=is_handwritten_page,
                            region_type=reg_type
                        )
                        if gem_res.get("verified"):
                            final_content = gem_res["text"]
                            final_conf = gem_res["confidence"]
                            method = gem_res["extraction_method"]
                            model_name_val = gem_res.get("model") or self.verifier.model_name
                            requires_review = gem_res["requires_review"]
                            gem_agreement = gem_res.get("verification_agreement")
                            gem_text = gem_res.get("text")
                            gem_conf_val = gem_res.get("confidence")
                            if gem_res.get("warnings"):
                                warnings.extend(gem_res.get("warnings", []))
                            gemini_handled = True
                            gemini_calls_count += 1
                    if not gemini_handled:
                        rec_res = self.recovery_engine.evaluate_and_recover(
                            crop=crop,
                            initial_text=reg.text,
                            initial_confidence=final_conf,
                            region_type=reg_type
                        )
                        final_content = rec_res.final_text
                        final_conf = rec_res.confidence
                        requires_review = rec_res.requires_review
                        alternatives = rec_res.alternatives
                        attempts_info = rec_res.attempts
                        raw_text_val = rec_res.raw_text
                        if rec_res.reason:
                            warnings.append(f"{rec_res.reason}: review required.")

            # 2b. SECTION REGION (Financial Statement Section Header)
            elif reg_type == "SECTION":
                b_type = BlockType.SECTION
                method = "financial_section_header"
                model_name_val = "financial_layout_analyzer"
                final_content = reg.text

            # 2c. HEADING / TITLE REGION
            elif reg_type == "HEADING":
                if reg.metadata.get("is_major_heading"):
                    b_type = BlockType.HEADING
                    method = "layout_classifier"
                    model_name_val = "layout_classifier"
                elif SemanticRegionDetector.is_amount(reg.text) or SemanticRegionDetector.is_financial_line_item(reg.text):
                    b_type = BlockType.LINE_ITEM
                    method = "financial_statement_row"
                    model_name_val = "financial_layout_analyzer"
                else:
                    b_type = BlockType.HEADING
                    method = "layout_classifier"
                    model_name_val = "layout_classifier"
                final_content = reg.text
                final_conf = max(final_conf, 0.85)

            # 3. LIST REGION
            elif reg_type == "LIST":
                b_type = BlockType.LIST_ITEM
                method = "layout_classifier"
                model_name_val = "layout_classifier"
                final_content = reg.text

            # 4. HEADER / FOOTER
            elif reg_type in ("HEADER", "FOOTER", "PAGE_NUMBER"):
                b_type = BlockType.HEADER if reg_type == "HEADER" else BlockType.FOOTER
                final_content = reg.text
                method = "layout_header_footer"
                model_name_val = "layout_header_footer"

            # 5. HANDWRITTEN TEXT / LOW CONFIDENCE (< GEMINI_REVIEW_THRESHOLD)
            elif (is_handwritten_page or final_conf < self.review_threshold) and crop is not None:
                gemini_handled = False

                # Primary enhancement attempt with Gemini Vision (within per-page budget)
                if self.verifier.is_available() and gemini_calls_count < max_gemini_calls:
                    gem_res = self.verifier.verify_and_enhance_block(
                        crop=crop,
                        ocr_text=reg.text,
                        bbox=scaled_bbox,
                        current_type=b_type.value,
                        current_conf=final_conf,
                        page_num=page_num,
                        is_handwritten=is_handwritten_page,
                        region_type=reg_type
                    )
                    if gem_res.get("verified"):
                        final_content = gem_res["text"]
                        final_conf = gem_res["confidence"]
                        method = gem_res["extraction_method"]
                        model_name_val = gem_res.get("model") or self.verifier.model_name
                        requires_review = gem_res["requires_review"]
                        gem_agreement = gem_res.get("verification_agreement")
                        gem_text = gem_res.get("text")
                        gem_conf_val = gem_res.get("confidence")
                        if gem_res.get("warnings"):
                            warnings.extend(gem_res.get("warnings", []))
                        if gem_res.get("is_equation") and gem_res.get("latex"):
                            b_type = BlockType.EQUATION
                            eq_data = EquationData(latex=gem_res["latex"], is_inline=False)
                        elif gem_res.get("type") == "heading":
                            if not SemanticRegionDetector.is_amount(gem_res["text"]) and not SemanticRegionDetector.is_financial_line_item(gem_res["text"]):
                                b_type = BlockType.HEADING
                            else:
                                b_type = BlockType.LINE_ITEM
                        elif gem_res.get("type") == "list_item":
                            b_type = BlockType.LIST_ITEM
                        gemini_handled = True
                        gemini_calls_count += 1

                # Fallback: if Gemini unavailable or failed, use local OCR / recovery
                if not gemini_handled:
                    if is_handwritten_page and final_conf < 0.80:
                        hw_res = self.hw_service.recognize_handwriting(crop, scaled_bbox)
                        if hw_res.get("text"):
                            final_content = hw_res["text"]
                            final_conf = hw_res["confidence"]
                            method = hw_res.get("model", "ocr_local")
                            model_name_val = "ocr_local"
                            if final_conf < 0.65:
                                requires_review = True
                                warnings.append("Handwriting recognition confidence below threshold.")
                    else:
                        # Low-confidence recovery loop
                        rec_res = self.recovery_engine.evaluate_and_recover(
                            crop=crop,
                            initial_text=reg.text,
                            initial_confidence=final_conf,
                            region_type=reg_type
                        )
                        final_content = rec_res.final_text
                        final_conf = rec_res.confidence
                        method = "ocr_local"
                        model_name_val = "ocr_local"
                        requires_review = rec_res.requires_review
                        alternatives = rec_res.alternatives
                        attempts_info = rec_res.attempts
                        raw_text_val = rec_res.raw_text
                        if rec_res.reason:
                            warnings.append(f"{rec_res.reason}: review required.")

            # Determine confidence level and status
            if final_conf >= 0.90:
                conf_level = ConfidenceLevel.HIGH
                status = BlockStatus.ACCEPTED
            elif final_conf >= self.review_threshold and not requires_review:
                conf_level = ConfidenceLevel.MEDIUM
                status = BlockStatus.ACCEPTED
            else:
                conf_level = ConfidenceLevel.LOW
                status = BlockStatus.REVIEW_REQUIRED
                requires_review = True
                if "Low extraction confidence; review required." not in warnings:
                    warnings.append("Low extraction confidence; review required.")

            block_metadata = {
                "region_type": reg_type,
                "requires_review": requires_review,
                "alternatives": alternatives,
                "model_name": model_name_val,
                "extraction_method": method,
                "original_ocr": reg.text,
                "ocr_confidence": round(reg.confidence, 4),
                "raw_text": locals().get("raw_text_val", reg.text),
                "final_text": final_content,
            }
            if reg.metadata:
                block_metadata.update(reg.metadata)
            if "confidence_breakdown" not in block_metadata:
                block_metadata["confidence_breakdown"] = {}
            if "gem_agreement" in locals() and gem_agreement is not None:
                block_metadata["verification_agreement"] = gem_agreement
                block_metadata["gemini_text"] = gem_text
                block_metadata["gemini_confidence"] = gem_conf_val
                block_metadata["confidence_breakdown"]["verification_agreement"] = gem_agreement
            if "rec_res" in locals() and getattr(rec_res, "retry_agreement", None) is not None:
                block_metadata["retry_agreement"] = rec_res.retry_agreement
                block_metadata["confidence_breakdown"]["retry_agreement"] = rec_res.retry_agreement
            if "attempts_info" in locals():
                block_metadata["attempts"] = attempts_info
                block_metadata["ocr_attempts"] = len(attempts_info)

            if b_type == BlockType.EQUATION and eq_data:
                eq_id = f"eq_p{page_num}_{idx+1:03d}"
                block_metadata["equation_id"] = eq_id
                block_metadata["latex"] = eq_data.latex
                block_metadata["raw_equation"] = reg.text

            block = SemanticBlock(
                block_id=block_id,
                type=b_type,
                content=final_content,
                source=SourceLocation(
                    file=filename,
                    page=page_num,
                    bbox=scaled_bbox,
                    coordinate_system="point"
                ),
                confidence=final_conf,
                confidence_level=conf_level,
                extraction_method=method,
                reading_order=idx + 1,
                status=status,
                warnings=warnings,
                equation_data=eq_data,
                table_data=tbl_data,
                chart_data=cht_data,
                metadata=block_metadata,
                model_name=model_name_val,
                requires_review=requires_review
            )
            blocks.append(block)

        # Apply multi-column reading order reconstruction
        ordered_blocks = ReadingOrderEngine.reconstruct_reading_order(
            blocks=blocks,
            page_width=page_w,
            page_height=page_h
        )

        return ordered_blocks
