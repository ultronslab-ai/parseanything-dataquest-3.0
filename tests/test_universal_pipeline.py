"""Comprehensive Integration Tests for ParseAnything Engine.
Verifies all format adapters (PDF, DOCX, PPTX, XLSX, IMAGE),
reading order, table merging, provenance, confidence, and fail-safe handling.
"""

import sys
from pathlib import Path

# Optional pytest fixture support
try:
    import pytest
    fixture_decorator = pytest.fixture
except ImportError:
    def fixture_decorator(f): return f

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.orchestrator import ExtractionPipeline
from schemas.models import BlockType, ConfidenceLevel
from schemas.errors import ErrorCode
from backend.cost_manager import CostManager


@pytest.fixture
def pipeline():
    return ExtractionPipeline()


def test_pdf_extraction(pipeline):
    path = BASE_DIR / "sample_document.pdf"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "success"
    assert doc.file_type == "pdf"
    assert len(doc.blocks) >= 4

    types = {b.type for b in doc.blocks}
    assert BlockType.HEADING in types or BlockType.PARAGRAPH in types
    assert BlockType.TABLE in types

    # Verify provenance
    for b in doc.blocks:
        assert b.source.file == "sample_document.pdf"
        assert b.source.page == 1
        assert b.source.bbox is not None
        assert 0.0 <= b.confidence <= 1.0


def test_docx_extraction(pipeline):
    path = BASE_DIR / "test_documents" / "08_word_report.docx"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "success"
    assert doc.file_type == "docx"
    types = {b.type for b in doc.blocks}
    assert BlockType.HEADING in types
    assert BlockType.TABLE in types


def test_pptx_extraction(pipeline):
    path = BASE_DIR / "test_documents" / "09_presentation_deck.pptx"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "success"
    assert doc.file_type == "pptx"
    assert any(b.source.slide is not None for b in doc.blocks)
    assert any(b.type == BlockType.TABLE for b in doc.blocks)


def test_xlsx_extraction(pipeline):
    path = BASE_DIR / "test_documents" / "10_financial_model.xlsx"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "success"
    assert doc.file_type == "xlsx"
    assert any(b.source.sheet is not None for b in doc.blocks)
    assert any(b.type == BlockType.TABLE for b in doc.blocks)


def test_image_ocr_extraction(pipeline):
    path = BASE_DIR / "test_documents" / "11_receipt_invoice.png"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "success"
    assert doc.file_type == "image"
    assert len(doc.blocks) > 0
    assert any(b.extraction_method.startswith("ocr") for b in doc.blocks)


def test_reading_order_two_column(pipeline):
    path = BASE_DIR / "test_documents" / "03_two_column_paper.pdf"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "success"
    # Reading order should be strictly sequential: 1, 2, 3, ...
    orders = [b.reading_order for b in doc.blocks]
    assert orders == list(range(1, len(doc.blocks) + 1))


def test_multipage_table_merging(pipeline):
    path = BASE_DIR / "test_documents" / "05_multipage_table.pdf"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "success"
    # Merged table should be present
    tables = [b for b in doc.blocks if b.type == BlockType.TABLE]
    assert len(tables) >= 1
    merged = [t for t in tables if t.table_data and t.table_data.is_multi_page_merged]
    assert len(merged) >= 1


def test_corrupt_file_fail_safe(pipeline):
    path = BASE_DIR / "test_documents" / "12_corrupt_file.pdf"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "failed"
    assert len(doc.errors) > 0
    assert doc.errors[0].code in (ErrorCode.CORRUPTED_FILE, ErrorCode.PARSER_FAILURE)


def test_unsupported_file_fail_safe(pipeline):
    path = BASE_DIR / "test_documents" / "13_unsupported_archive.bin"
    assert path.exists()
    doc = pipeline.process(path)

    assert doc.processing_status == "failed"
    assert len(doc.errors) > 0
    assert doc.errors[0].code == ErrorCode.UNSUPPORTED_FORMAT


def test_cost_manager():
    cm = CostManager()
    cm.record_call("native_extractor", pages=100)
    cm.record_call("local_ocr", pages=50)

    summary = cm.get_metrics_summary()
    assert summary["total_estimated_cost_usd"] == 0.0
    assert summary["is_within_budget"] is True
    assert summary["estimated_cost_per_1000_pages_usd"] == 0.0


if __name__ == "__main__":
    p = ExtractionPipeline()
    print("Running integration tests...")
    test_pdf_extraction(p)
    print("-> PDF extraction: OK")
    test_docx_extraction(p)
    print("-> DOCX extraction: OK")
    test_pptx_extraction(p)
    print("-> PPTX extraction: OK")
    test_xlsx_extraction(p)
    print("-> XLSX extraction: OK")
    test_image_ocr_extraction(p)
    print("-> Image OCR extraction: OK")
    test_reading_order_two_column(p)
    print("-> Two-column reading order: OK")
    test_multipage_table_merging(p)
    print("-> Multi-page table merge: OK")
    test_corrupt_file_fail_safe(p)
    print("-> Corrupt file fail-safe: OK")
    test_unsupported_file_fail_safe(p)
    print("-> Unsupported file fail-safe: OK")
    test_cost_manager()
    print("-> Cost manager: OK")
    print("\nALL 10 INTEGRATION TESTS PASSED PERFECTLY!")
