"""Comprehensive test script for Markdown Generation across:
1. Scanned Image-Only Financial Statement PDF (Balance Sheet & Income Statement)
2. Normal digital PDF with tables
3. Multi-page document
4. Notes with LaTeX equations
"""

import sys
from pathlib import Path

# Fix console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipeline.orchestrator import ExtractionPipeline


def run_tests():
    pipeline = ExtractionPipeline()
    backend_data = BASE_DIR / "backend" / "data"

    test_files = {
        "1. Scanned Financial PDF": backend_data / "doc_b2b733ec_Sample-Financial-Statements-image-only.pdf",
        "2. Digital PDF with Tables": backend_data / "test_digital_sample.pdf",
        "3. Multi-Page Document": backend_data / "doc_4f0c1fb2_Plant_Location_Theory_Summary.pdf",
        "4. Document with Equations": backend_data / "temp_test_1p.pdf"
    }

    for label, file_path in test_files.items():
        if not file_path.exists():
            print(f"[SKIP] {label}: {file_path} not found")
            continue

        print("\n" + "=" * 70)
        print(f"TESTING: {label} ({file_path.name})")
        print("=" * 70)

        doc = pipeline.process(file_path, document_id=f"test_run_{file_path.stem[:8]}")
        md = doc.markdown or ""

        print(f"Total Blocks Extracted: {len(doc.blocks)}")
        print(f"Markdown Character Length: {len(md)}")
        print(f"Markdown Lines: {len(md.splitlines())}")
        print("\n--- RENDERED MARKDOWN PREVIEW ---")
        print(md[:1500])
        print("---------------------------------")

        # Specific assertion checks
        assert len(md) > 0, f"Markdown for {label} must not be empty"
        assert "*Source: Page" not in md, f"Markdown should not contain debug source tags"
        assert "⚠️ **Review required" not in md, f"Markdown should not contain warning banners"

        # Check for financial statement assertions
        if "Financial" in label:
            # Check that numeric amounts are NOT headings:
            assert "## $13,060" not in md, "Numeric amount $13,060 must NOT become a heading"
            assert "## 4.165" not in md, "Numeric amount 4.165 must NOT become a heading"
            assert "## 100" not in md, "Numeric amount 100 must NOT become a heading"
            assert "## 1,200" not in md, "Numeric amount 1,200 must NOT become a heading"
            assert "## 8,000" not in md, "Numeric amount 8,000 must NOT become a heading"
            assert "## Total Current Liabilities" not in md, "Total Current Liabilities must NOT be a heading"
            assert "## Owners' capital" not in md, "Owners' capital must NOT be a heading"

            # Check that genuine headings exist:
            assert "# Sample Company" in md, "Should preserve genuine company heading"
            assert "## Balance Sheet" in md, "Should preserve genuine Balance Sheet heading"
            assert "| Assets | Amount | Liabilities and Stockholders' Equity | Amount |" in md, "Should contain structured Balance Sheet table"

    print("\n" + "=" * 70)
    print("ALL TESTS AND ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
