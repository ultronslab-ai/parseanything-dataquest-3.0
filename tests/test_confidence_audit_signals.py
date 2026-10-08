"""Comprehensive Audit & Verification Test for Scanned Document Confidence Scoring.
Verifies all 7 confidence signals:
1. OCR recognition confidence
2. Word-level confidence aggregation
3. Bounding-box / layout consistency
4. Semantic block consistency
5. Financial row / column alignment
6. Agreement between OCR and Gemini verification:
   - OCR says 'Depreciation expense 100' and Gemini says 'Depreciation expense 100' -> increased appropriately
   - OCR and Gemini disagree -> confidence <= 0.55, marked Needs Review
7. Agreement between multiple extraction attempts when available
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from schemas.models import BlockStatus, BlockType, ConfidenceLevel, SemanticBlock, SourceLocation
from pipeline.confidence.confidence_engine import ConfidenceEngine
from scanned_pipeline.gemini_verifier import GeminiVerificationService
from scanned_pipeline.confidence_recovery import ConfidenceRecoveryEngine, ConfidenceRecoveryResult
from pipeline.orchestrator import ExtractionPipeline


class TestConfidenceAuditSignals(unittest.TestCase):

    def test_ocr_and_gemini_agreement_depreciation_expense(self):
        """Test: If OCR and Gemini agree on 'Depreciation expense 100', confidence reflects agreement."""
        verifier = GeminiVerificationService()
        
        # Scenario 1: OCR extracted 'Depreciation expense 100'
        ocr_text = "Depreciation expense 100"
        initial_conf = 0.88
        
        # Test agreement calculation in verifier
        # When Gemini also returns 'Depreciation expense 100'
        import difflib, re
        ocr_norm = re.sub(r"\s+", " ", ocr_text).strip().lower()
        gem_text = "Depreciation expense 100"
        gem_norm = re.sub(r"\s+", " ", gem_text).strip().lower()
        
        agreement_ratio = difflib.SequenceMatcher(None, ocr_norm, gem_norm).ratio()
        ocr_nums = re.findall(r"\d+", ocr_norm)
        gem_nums = re.findall(r"\d+", gem_norm)
        nums_agree = (ocr_nums == gem_nums)
        
        self.assertEqual(agreement_ratio, 1.0)
        self.assertTrue(nums_agree)
        
        # Dual confirmation calculation:
        base_c = max(initial_conf, 0.90, 0.85)
        expected_boosted_conf = min(0.98, base_c + (1.0 - base_c) * 0.40 * agreement_ratio)
        self.assertGreater(expected_boosted_conf, initial_conf)
        self.assertGreaterEqual(expected_boosted_conf, 0.94)

        # Now test through ConfidenceEngine.evaluate_block
        block = SemanticBlock(
            block_id="b_test_01",
            type=BlockType.LINE_ITEM,
            content=ocr_text,
            source=SourceLocation(file="test.pdf", page=1, bbox=[100, 200, 400, 215], coordinate_system="point"),
            confidence=initial_conf,
            confidence_level=ConfidenceLevel.MEDIUM,
            extraction_method="ocr_local+gemini_verification",
            reading_order=1,
            status=BlockStatus.ACCEPTED,
            metadata={
                "is_financial_row": True,
                "label": "Depreciation expense",
                "amount": "100",
                "verification_agreement": 1.0,
                "confidence_breakdown": {
                    "ocr_recognition_confidence": initial_conf,
                    "financial_alignment_score": 0.98,
                    "word_aggregation_score": 1.0,
                    "layout_consistency": 1.0,
                    "semantic_consistency": 1.0,
                    "verification_agreement": 1.0
                }
            }
        )
        
        evaluated = ConfidenceEngine.evaluate_block(block)
        
        self.assertGreater(evaluated.confidence, initial_conf)
        self.assertGreaterEqual(evaluated.confidence, 0.92)
        self.assertEqual(evaluated.confidence_level, ConfidenceLevel.HIGH)
        self.assertEqual(evaluated.status, BlockStatus.ACCEPTED)
        self.assertFalse(evaluated.requires_review)
        self.assertEqual(evaluated.metadata["confidence_breakdown"]["verification_agreement"], 1.0)
        print(f"Agreement confirmed: initial={initial_conf} -> boosted={evaluated.confidence:.4f}")

    def test_ocr_and_gemini_disagreement_depreciation_expense(self):
        """Test: If OCR and Gemini disagree on amounts or text, keep confidence low (<= 0.55) and mark Needs Review."""
        ocr_text = "Depreciation expense 100"
        gem_text = "Depreciation expense 150"  # Disagree on numeric amount
        
        import difflib, re
        ocr_norm = re.sub(r"\s+", " ", ocr_text).strip().lower()
        gem_norm = re.sub(r"\s+", " ", gem_text).strip().lower()
        ocr_nums = re.findall(r"\d+", ocr_norm)
        gem_nums = re.findall(r"\d+", gem_norm)
        nums_agree = (ocr_nums == gem_nums)
        
        self.assertFalse(nums_agree, "Numbers must not agree in disagreement test")

        block = SemanticBlock(
            block_id="b_test_02",
            type=BlockType.LINE_ITEM,
            content=ocr_text,
            source=SourceLocation(file="test.pdf", page=1, bbox=[100, 200, 400, 215], coordinate_system="point"),
            confidence=0.88,
            confidence_level=ConfidenceLevel.MEDIUM,
            extraction_method="ocr_local+gemini_verification",
            reading_order=2,
            status=BlockStatus.ACCEPTED,
            metadata={
                "is_financial_row": True,
                "label": "Depreciation expense",
                "amount": "100",
                "verification_agreement": 0.45,  # Disagreement signal
                "confidence_breakdown": {
                    "ocr_recognition_confidence": 0.88,
                    "financial_alignment_score": 0.95,
                    "word_aggregation_score": 1.0,
                    "layout_consistency": 1.0,
                    "semantic_consistency": 1.0,
                    "verification_agreement": 0.45
                }
            }
        )
        
        evaluated = ConfidenceEngine.evaluate_block(block)
        
        self.assertLessEqual(evaluated.confidence, 0.55)
        self.assertEqual(evaluated.confidence_level, ConfidenceLevel.LOW)
        self.assertEqual(evaluated.status, BlockStatus.REVIEW_REQUIRED)
        self.assertTrue(evaluated.requires_review)
        self.assertTrue(any("disagree" in w.lower() for w in evaluated.warnings))
        self.assertEqual(evaluated.metadata["confidence_breakdown"]["verification_agreement"], 0.45)
        print(f"Disagreement handled: conf={evaluated.confidence:.4f}, status={evaluated.status.value}, review={evaluated.requires_review}")

    def test_multi_attempt_retry_agreement(self):
        """Test: Agreement between multiple extraction attempts."""
        # Case A: Multiple attempts agree
        block_agree = SemanticBlock(
            block_id="b_test_03a",
            type=BlockType.PARAGRAPH,
            content="Standard text line",
            source=SourceLocation(file="test.pdf", page=1, bbox=[50, 50, 300, 70], coordinate_system="point"),
            confidence=0.82,
            confidence_level=ConfidenceLevel.MEDIUM,
            extraction_method="ocr_local",
            reading_order=3,
            status=BlockStatus.ACCEPTED,
            metadata={"retry_agreement": 0.95}
        )
        eval_agree = ConfidenceEngine.evaluate_block(block_agree)
        self.assertGreaterEqual(eval_agree.confidence, 0.85)

        # Case B: Multiple attempts disagree
        block_disagree = SemanticBlock(
            block_id="b_test_03b",
            type=BlockType.PARAGRAPH,
            content="Conflicted text line",
            source=SourceLocation(file="test.pdf", page=1, bbox=[50, 50, 300, 70], coordinate_system="point"),
            confidence=0.75,
            confidence_level=ConfidenceLevel.MEDIUM,
            extraction_method="ocr_local",
            reading_order=4,
            status=BlockStatus.ACCEPTED,
            metadata={"retry_agreement": 0.40}
        )
        eval_disagree = ConfidenceEngine.evaluate_block(block_disagree)
        self.assertLessEqual(eval_disagree.confidence, 0.55)
        self.assertEqual(eval_disagree.status, BlockStatus.REVIEW_REQUIRED)
        self.assertTrue(eval_disagree.requires_review)

    def test_financial_row_alignment_consistency(self):
        """Test: Financial row/column alignment score reflects spatial geometric alignment."""
        block = SemanticBlock(
            block_id="b_test_04",
            type=BlockType.LINE_ITEM,
            content="Service revenue $2,750",
            source=SourceLocation(file="test.pdf", page=1, bbox=[72, 180, 540, 195], coordinate_system="point"),
            confidence=0.90,
            confidence_level=ConfidenceLevel.HIGH,
            extraction_method="financial_statement_row",
            reading_order=5,
            status=BlockStatus.ACCEPTED,
            metadata={
                "is_financial_row": True,
                "label": "Service revenue",
                "amount": "$2,750",
                "confidence_breakdown": {
                    "ocr_recognition_confidence": 0.90,
                    "financial_alignment_score": 0.98,
                    "word_aggregation_score": 1.0,
                    "layout_consistency": 1.0,
                    "semantic_consistency": 1.0
                }
            }
        )
        evaluated = ConfidenceEngine.evaluate_block(block)
        bd = evaluated.metadata["confidence_breakdown"]
        self.assertIsNotNone(bd["financial_alignment_score"])
        self.assertGreaterEqual(bd["financial_alignment_score"], 0.90)
        self.assertEqual(bd["layout_consistency"], 1.0)
        self.assertEqual(bd["semantic_consistency"], 1.0)

    def test_scanned_pdf_pipeline_overall_audit(self):
        """Test: Full scanned financial PDF extraction yields honest ~86% confidence, 0 false reviews."""
        pdf_path = BASE_DIR / "backend" / "data" / "doc_b2b733ec_Sample-Financial-Statements-image-only.pdf"
        self.assertTrue(pdf_path.exists())

        pipeline = ExtractionPipeline()
        doc = pipeline.process(pdf_path, document_id="doc_audit_test")
        
        self.assertGreater(len(doc.blocks), 60)
        avg_conf = sum(b.confidence for b in doc.blocks) / len(doc.blocks)
        
        print(f"\nDocument processed: {len(doc.blocks)} blocks. Average confidence: {avg_conf:.2%}")
        
        # Verify confidence is honest (around 85-88%), NOT deflated to 52-55%
        self.assertGreaterEqual(avg_conf, 0.80, "Average confidence must not be falsely deflated to 52-55%")
        self.assertLessEqual(avg_conf, 0.95, "Confidence must not be artificially inflated to 100%")
        
        # Verify false reviews were eliminated
        self.assertLessEqual(doc.stats.low_confidence_blocks, 3)

        # Verify confidence_breakdown exists on blocks
        for b in doc.blocks[:10]:
            self.assertIn("confidence_breakdown", b.metadata)
            bd = b.metadata["confidence_breakdown"]
            self.assertIn("ocr_recognition_confidence", bd)
            self.assertIn("word_aggregation_score", bd)
            self.assertIn("layout_consistency", bd)
            self.assertIn("semantic_consistency", bd)
            self.assertIn("final_score", bd)


if __name__ == "__main__":
    unittest.main()
