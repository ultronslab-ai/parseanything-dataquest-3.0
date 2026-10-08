from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pymupdf
from pipeline.orchestrator import ExtractionPipeline

# Extract page 1 to a single page PDF
src = Path("backend/data/doc_f0860f29_EMT Unit-2 Notes & problems.pdf")
if not src.exists():
    print("EMT file not found.")
    exit(0)

doc_src = pymupdf.open(str(src))
single_pdf_path = Path("backend/data/test_emt_p1.pdf")
single_doc = pymupdf.open()
single_doc.insert_pdf(doc_src, from_page=0, to_page=0)
single_doc.save(str(single_pdf_path))
single_doc.close()
doc_src.close()

p = ExtractionPipeline()

# Test with ENABLE_SCANNED_FALLBACK = True (New pipeline)
os.environ["ENABLE_SCANNED_FALLBACK"] = "true"
doc_new = p.process(single_pdf_path, document_id="test_new")
print("NEW SCANNED PIPELINE:")
print("  Page Type:", doc_new.pages[0].page_type)
print("  Block count:", len(doc_new.blocks))
avg_conf_new = round(sum(b.confidence for b in doc_new.blocks)/max(1, len(doc_new.blocks)), 3)
print("  Average confidence:", avg_conf_new)
print("  First 4 blocks:")
for i, b in enumerate(doc_new.blocks[:4]):
    print(f"    Block {i+1} [{b.type}]: '{b.content[:70]}' (conf: {b.confidence}, method: {b.extraction_method})")

# Test with ENABLE_SCANNED_FALLBACK = False (Baseline pipeline)
os.environ["ENABLE_SCANNED_FALLBACK"] = "false"
doc_old = p.process(single_pdf_path, document_id="test_old")
print("\nBASELINE PIPELINE (Before):")
print("  Page Type:", doc_old.pages[0].page_type)
print("  Block count:", len(doc_old.blocks))
avg_conf_old = round(sum(b.confidence for b in doc_old.blocks)/max(1, len(doc_old.blocks)), 3)
print("  Average confidence:", avg_conf_old)
print("  First 4 blocks:")
for i, b in enumerate(doc_old.blocks[:4]):
    print(f"    Block {i+1} [{b.type}]: '{b.content[:70]}' (conf: {b.confidence}, method: {b.extraction_method})")
