import sys
import os
import re
from pathlib import Path

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import fitz
from PIL import Image
from schemas.models import BlockType, Document, DocumentMetadata, ProcessingStats, PageRenderInfo
from scanned_pipeline.scanned_orchestrator import ScannedDocumentPipeline
from pipeline.assembly.markdown_generator import MarkdownGenerator

def test_financial_image_only_pdf():
    print("=" * 70)
    print("TEST: Scanned Image-Only Financial PDF Extraction & Block Grouping")
    print("=" * 70)

    pdf_path = BASE_DIR / "backend" / "data" / "doc_b2b733ec_Sample-Financial-Statements-image-only.pdf"
    assert pdf_path.exists(), f"File not found: {pdf_path}"

    doc = fitz.open(str(pdf_path))
    pipeline = ScannedDocumentPipeline()

    all_blocks = []
    render_infos = []

    for pno in range(len(doc)):
        page = doc[pno]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        result = pipeline.process_page(
            image=img,
            page_num=pno + 1,
            filename=pdf_path.name,
            page_w=page.rect.width,
            page_h=page.rect.height
        )
        assert result.success, f"Page {pno+1} failed: {result.error}"
        all_blocks.extend(result.blocks)
        render_infos.append(PageRenderInfo(
            page_number=pno + 1,
            width=page.rect.width,
            height=page.rect.height
        ))
        
        print(f"\nPage {pno+1}: {len(result.blocks)} semantic blocks reconstructed")
        for b in result.blocks:
            lbl = b.metadata.get("label", "")
            amt = b.metadata.get("amount", "")
            meta_str = f" [label={repr(lbl)}, amt={repr(amt)}]" if (lbl or amt) else ""
            print(f"  #{b.reading_order:02d} [{b.type.value:<10}] (conf={b.confidence:.2f}, rev={b.requires_review}): {repr(b.content)}{meta_str}")

    print("\n--- ASSERTION CHECKS ---")

    # 1. Check: NUMBERS MUST NOT BECOME HEADINGS
    bad_number_headings = []
    for b in all_blocks:
        if b.type == BlockType.HEADING:
            clean = b.content.strip().replace("$", "").replace(",", "").replace(".", "").replace("(", "").replace(")", "")
            if clean.isdigit():
                bad_number_headings.append(b.content)

    print(f"Bad numeric headings count: {len(bad_number_headings)}")
    assert len(bad_number_headings) == 0, f"Found numeric headings: {bad_number_headings}"

    # 2. Check: FINANCIAL LINE ITEMS MUST NOT BE AUTOMATICALLY CLASSIFIED AS HEADINGS
    line_item_terms = ["depreciation expense", "wages expenses", "supplies expenses", "total operating expenses", "accounts payable", "interest payable", "owners' capital"]
    bad_line_item_headings = []
    for b in all_blocks:
        if b.type == BlockType.HEADING:
            for term in line_item_terms:
                if term in b.content.lower():
                    bad_line_item_headings.append(b.content)

    print(f"Bad line item headings count: {len(bad_line_item_headings)}")
    assert len(bad_line_item_headings) == 0, f"Found line items classified as headings: {bad_line_item_headings}"

    # 3. Check: TRUE HEADINGS EXIST
    heading_contents = [b.content for b in all_blocks if b.type == BlockType.HEADING]
    print(f"Genuine headings detected: {heading_contents}")
    assert any("Sample Company" in h for h in heading_contents), "Sample Company heading missing"
    assert any("Income Statement" in h for h in heading_contents), "Income Statement heading missing"
    assert any("Balance Sheet" in h for h in heading_contents), "Balance Sheet heading missing"

    # 4. Check: LINE_ITEM BLOCKS EXIST WITH ASSOCIATED LABELS & AMOUNTS
    line_items = [b for b in all_blocks if b.type == BlockType.LINE_ITEM]
    print(f"LINE_ITEM blocks detected: {len(line_items)}")
    assert len(line_items) >= 20, f"Expected at least 20 LINE_ITEM blocks, found {len(line_items)}"

    # 5. Check: CONFIDENCE SCORES ARE HONEST AND NOT FALSELY PEGGED LOW (< 0.60)
    low_conf_blocks = [b for b in all_blocks if b.confidence < 0.65]
    print(f"Blocks with conf < 0.65: {len(low_conf_blocks)}")
    assert len(low_conf_blocks) <= 2, f"Too many low confidence blocks ({len(low_conf_blocks)}): {[b.content for b in low_conf_blocks]}"

    # 6. Test MARKDOWN GENERATION FROM THE SEMANTIC BLOCKS
    test_doc = Document(
        document_id="doc_test_financial",
        filename=pdf_path.name,
        file_type="pdf",
        file_size_bytes=pdf_path.stat().st_size,
        blocks=all_blocks,
        pages=render_infos,
        metadata=DocumentMetadata(page_count=len(doc)),
        stats=ProcessingStats(pages_processed=len(doc), blocks_extracted=len(all_blocks))
    )
    md = MarkdownGenerator.generate(test_doc)
    print("\n--- RENDERED MARKDOWN FULL ---")
    print(md)
    print("---------------------------------")

    assert "# Sample Company" in md
    assert "## Balance Sheet" in md
    assert "| Assets | Amount | Liabilities and Stockholders' Equity | Amount |" in md
    assert "1,550" in md
    assert "13,060" in md or "13.060" in md
    print("\nALL FINANCIAL SEMANTIC GROUPING CHECKS PASSED!")

if __name__ == "__main__":
    test_financial_image_only_pdf()
