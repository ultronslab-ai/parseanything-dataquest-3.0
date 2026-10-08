"""Automated Tests for Document Extraction Pipeline."""

from pathlib import Path
from doc_extractor.models import ElementType
from doc_extractor.pipeline import DocumentExtractionPipeline


def test_extracted_elements_schema():
    sample_pdf = Path(__file__).parent.parent / "sample_document.pdf"
    assert sample_pdf.exists(), f"Sample PDF must exist at {sample_pdf}"

    pipeline = DocumentExtractionPipeline(engine="hybrid", enable_visualization=False)
    result = pipeline.process(file_path=sample_pdf, export_json=False)

    assert result.total_pages >= 1
    assert len(result.elements) > 0

    found_types = set()

    for elem in result.elements:
        # Check required fields
        assert elem.content is not None and len(str(elem.content).strip()) > 0, "Content must not be empty"
        assert elem.bbox is not None, "Bbox must be present"
        assert elem.bbox.x_min <= elem.bbox.x_max, "x_min must be <= x_max"
        assert elem.bbox.y_min <= elem.bbox.y_max, "y_min must be <= y_max"

        assert elem.source is not None, "Source must be present"
        assert elem.source.page_number >= 1, "Page number must be >= 1"
        assert elem.source.document_name == "sample_document.pdf"

        assert 0.0 <= elem.confidence <= 1.0, f"Confidence must be between 0.0 and 1.0, got {elem.confidence}"

        found_types.add(elem.type)

    # Verify that all 4 requested element types are present
    assert ElementType.TEXT in found_types, "Text elements should be detected"
    assert ElementType.TABLE in found_types, "Table elements should be detected"
    assert ElementType.CHART in found_types, "Chart elements should be detected"
    assert ElementType.EQUATION in found_types, "Equation elements should be detected"

    print("\n[OK] All schema assertions passed successfully!")
    print(f"[OK] Detected types: {[t.value for t in found_types]}")


if __name__ == "__main__":
    test_extracted_elements_schema()
