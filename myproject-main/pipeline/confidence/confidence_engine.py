"""Confidence Engine for ParseAnything.
Calculates transparent confidence scores based on multi-factor signals:
1. OCR recognition confidence
2. Word-level confidence aggregation
3. Bounding-box and layout consistency
4. Semantic block consistency
5. Financial row/column alignment
6. Agreement between OCR and Gemini verification
7. Agreement between multiple extraction attempts when available

Assigns HIGH (>= 0.90), MEDIUM (0.70 - 0.89), and LOW (< 0.70).
Sets status = 'review_required' for low-confidence blocks.
Preserves confidence breakdown and full provenance coordinates.
"""

import re
from typing import Any, Dict, List, Optional
from schemas.models import BlockStatus, BlockType, ConfidenceLevel, SemanticBlock


class ConfidenceEngine:
    """Evaluates and updates block confidence using transparent signals."""

    HIGH_THRESHOLD = 0.90
    MEDIUM_THRESHOLD = 0.70

    @classmethod
    def evaluate_block(cls, block: SemanticBlock) -> SemanticBlock:
        """Computes and assigns final confidence score, level, and review flags."""
        raw_score = float(block.confidence)
        base_ocr_score = raw_score
        text = (block.content or "").strip()
        warnings = list(block.warnings)

        # -----------------------------------------------------------------
        # SIGNAL 1 & 2: OCR Recognition & Word-Level Confidence Aggregation
        # -----------------------------------------------------------------
        word_agg_score = 1.0
        if text:
            # Check ratio of printable characters
            printable_ratio = sum(c.isprintable() for c in text) / len(text)
            if printable_ratio < 0.85:
                raw_score *= 0.75
                warnings.append("High ratio of unprintable or garbled characters.")
                word_agg_score *= 0.75

            # Check for genuine replacement bytes or corrupt escapes (e.g. \ufffd or raw \x hex escapes)
            if "\ufffd" in text or "\\x" in text:
                raw_score *= 0.60
                warnings.append("Text contains corrupt replacement bytes.")
                word_agg_score *= 0.60

        # -----------------------------------------------------------------
        # SIGNAL 3: Bounding-Box & Layout Consistency
        # -----------------------------------------------------------------
        layout_score = 1.0
        bbox = block.source.bbox if block.source else None
        if bbox:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 0 or h <= 0:
                raw_score *= 0.50
                warnings.append("Degenerate or inverted bounding box dimensions.")
                layout_score = 0.50
            elif bbox[0] < -10 or bbox[1] < -10:
                raw_score *= 0.70
                warnings.append("Bounding box coordinates outside valid canvas.")
                layout_score = 0.70

        # -----------------------------------------------------------------
        # SIGNAL 4: Semantic Block Consistency
        # -----------------------------------------------------------------
        semantic_score = 1.0
        if block.type == BlockType.HEADING:
            words = text.split()
            # Headings must never be isolated numbers
            if len(words) == 1 and any(c.isdigit() for c in words[0]) and not any(c.isalpha() for c in words[0]):
                raw_score = min(raw_score, 0.45)
                warnings.append("Isolated numeric value classified as heading.")
                semantic_score = 0.45
            elif len(words) > 20:
                raw_score *= 0.85
                semantic_score = 0.85
        elif block.type == BlockType.TABLE and block.table_data:
            td = block.table_data
            if td.num_cols == 0 or td.num_rows == 0:
                raw_score = 0.50
                warnings.append("Empty table dimensions detected.")
                semantic_score = 0.50
            else:
                filled_cells = sum(1 for c in td.cells if c.text.strip())
                total_cells = max(1, len(td.cells))
                fill_ratio = filled_cells / total_cells
                if fill_ratio < 0.30:
                    raw_score *= 0.80
                    warnings.append("Table contains over 70% empty cells.")
                    semantic_score *= 0.80
        elif block.type == BlockType.CHART and block.chart_data:
            if block.chart_data.status == "partial":
                raw_score = min(0.85, raw_score)
                semantic_score = 0.85

        # -----------------------------------------------------------------
        # SIGNAL 5: Extraction Method Base Confidence & Financial Row Alignment
        # -----------------------------------------------------------------
        method = (block.extraction_method or "").lower()
        align_score = None
        if "native" in method:
            raw_score = max(0.92, raw_score)
        elif "financial" in method or block.metadata.get("is_financial_row"):
            meta_bd = block.metadata.get("confidence_breakdown", {})
            align_score = meta_bd.get("financial_alignment_score")
            if align_score is None:
                align_score = 0.95 if (block.metadata.get("label") and block.metadata.get("amount")) else 0.80
            # If line item has both label and amount with baseline alignment:
            if block.metadata.get("label") and block.metadata.get("amount"):
                raw_score = max(raw_score, meta_bd.get("composite_confidence", raw_score))

        # -----------------------------------------------------------------
        # SIGNAL 6: Agreement Between OCR & Gemini Verification
        # -----------------------------------------------------------------
        verif_agreement = None
        if "gemini" in method or block.metadata.get("verification_agreement") is not None:
            verif_agreement = block.metadata.get("verification_agreement", 1.0)
            if verif_agreement >= 0.85:
                # OCR and Gemini agree on text
                raw_score = min(0.98, max(raw_score, 0.88) + 0.05)
            elif verif_agreement < 0.65:
                # Disagreement between OCR and Gemini
                raw_score = min(raw_score, 0.55)
                warnings.append("OCR and Gemini verification disagree on content.")

        # -----------------------------------------------------------------
        # SIGNAL 7: Agreement Between Multiple Extraction Attempts
        # -----------------------------------------------------------------
        retry_agreement = block.metadata.get("retry_agreement")
        if retry_agreement is not None:
            if retry_agreement >= 0.90:
                raw_score = min(0.95, raw_score + 0.05)
            elif retry_agreement < 0.60:
                raw_score = min(raw_score, 0.55)
                warnings.append("Multi-pass OCR extraction attempts produced conflicting candidates.")

        final_score = round(max(0.0, min(1.0, raw_score)), 4)
        block.confidence = final_score
        block.warnings = warnings

        # Determine Confidence Level & Review Status
        requires_review = block.requires_review or False
        if final_score >= cls.HIGH_THRESHOLD and not requires_review:
            block.confidence_level = ConfidenceLevel.HIGH
            block.status = BlockStatus.ACCEPTED
        elif final_score >= cls.MEDIUM_THRESHOLD and not requires_review:
            block.confidence_level = ConfidenceLevel.MEDIUM
            block.status = BlockStatus.ACCEPTED
        else:
            block.confidence_level = ConfidenceLevel.LOW
            block.status = BlockStatus.REVIEW_REQUIRED
            block.requires_review = True
            if "Low extraction confidence; review required." not in block.warnings:
                block.warnings.append("Low extraction confidence; review required.")

        # Save transparent confidence breakdown into metadata
        block.metadata.setdefault("confidence_breakdown", {})
        block.metadata["confidence_breakdown"].update({
            "ocr_recognition_confidence": round(block.metadata.get("ocr_confidence", base_ocr_score), 4),
            "word_aggregation_score": round(word_agg_score, 4),
            "layout_consistency": round(layout_score, 4),
            "semantic_consistency": round(semantic_score, 4),
            "financial_alignment_score": round(align_score, 4) if align_score is not None else None,
            "verification_agreement": round(verif_agreement, 4) if verif_agreement is not None else None,
            "retry_agreement": round(retry_agreement, 4) if retry_agreement is not None else None,
            "final_score": final_score
        })

        return block

    @classmethod
    def evaluate_all(cls, blocks: List[SemanticBlock]) -> List[SemanticBlock]:
        """Evaluates all blocks in document."""
        return [cls.evaluate_block(b) for b in blocks]
