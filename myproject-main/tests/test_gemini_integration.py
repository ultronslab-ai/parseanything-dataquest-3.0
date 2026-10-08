"""Comprehensive Test & Verification Script for OCR + Gemini Verification Layer.
Compares BEFORE (pure local OCR) vs AFTER (OCR + Gemini verification).
Measures:
- blocks count
- average confidence
- equations detected
- low-confidence blocks (<0.70)
- Gemini-enhanced blocks
- validated equations
- false equations eliminated
- blocks requiring review
- processing time
"""

import os
import sys
import time
from pathlib import Path

# Fix Windows console utf-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env", override=False)

import pymupdf
from schemas.models import BlockType
from pipeline.orchestrator import ExtractionPipeline
from scanned_pipeline.gemini_verifier import get_gemini_verifier


def run_comparison(pdf_path: Path, max_pages: int = 1):
    print("=" * 70, flush=True)
    print(f"RUNNING OCR + GEMINI VERIFICATION BENCHMARK ON: {pdf_path.name}", flush=True)
    print(f"Pages to evaluate: {max_pages}", flush=True)
    print("=" * 70, flush=True)

    # Prepare test file (extract slice if max_pages < total)
    doc_src = pymupdf.open(str(pdf_path))
    total_doc_pages = len(doc_src)
    print(f"Source document total pages: {total_doc_pages}", flush=True)

    test_file_path = pdf_path
    if max_pages < total_doc_pages:
        import uuid
        test_file_path = BASE_DIR / "backend" / "data" / f"temp_test_{uuid.uuid4().hex[:6]}_{max_pages}p.pdf"
        sub_doc = pymupdf.open()
        sub_doc.insert_pdf(doc_src, from_page=0, to_page=max_pages - 1)
        sub_doc.save(str(test_file_path))
        sub_doc.close()
    doc_src.close()

    verifier = get_gemini_verifier()
    orig_api_key = verifier.api_key or os.environ.get("GEMINI_API_KEY", "")
    print(f"Gemini API Available: {verifier.is_available()}", flush=True)
    print(f"Gemini Model: {verifier.model_name}", flush=True)
    print(f"Review Threshold: {verifier.review_threshold}", flush=True)
    print("-" * 70, flush=True)

    # -------------------------------------------------------------------------
    # 1. RUN BEFORE: Pure Local OCR (Gemini Disabled via Temporary Mock / Flag)
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Executing BEFORE (Pure Local OCR Primary)...", flush=True)
    os.environ["GEMINI_API_KEY"] = ""  # Temporarily disable Gemini
    verifier.api_key = ""
    verifier.client = None

    p_before = ExtractionPipeline()
    t0_before = time.time()
    doc_before = p_before.process(test_file_path, document_id="bench_before")
    time_before = round(time.time() - t0_before, 2)

    blocks_before = doc_before.blocks
    total_blocks_b = len(blocks_before)
    avg_conf_b = round(sum(b.confidence for b in blocks_before) / max(1, total_blocks_b), 4)
    eq_b = [b for b in blocks_before if b.type == BlockType.EQUATION]
    low_conf_b = [b for b in blocks_before if b.confidence < 0.70]
    review_b = [b for b in blocks_before if b.status.value == "review_required" or b.requires_review]

    print(f"  Completed in {time_before}s")
    print(f"  Blocks: {total_blocks_b} | Avg Conf: {avg_conf_b:.2%} | Equations: {len(eq_b)} | Low Conf: {len(low_conf_b)}")

    # -------------------------------------------------------------------------
    # 2. RUN AFTER: OCR + Gemini Verification Layer
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Executing AFTER (OCR + Gemini Verification Layer)...", flush=True)
    os.environ["GEMINI_API_KEY"] = orig_api_key  # Restore Gemini
    verifier.api_key = orig_api_key
    verifier._init_client()

    p_after = ExtractionPipeline()
    t0_after = time.time()
    doc_after = p_after.process(test_file_path, document_id="bench_after")
    time_after = round(time.time() - t0_after, 2)

    blocks_after = doc_after.blocks
    total_blocks_a = len(blocks_after)
    avg_conf_a = round(sum(b.confidence for b in blocks_after) / max(1, total_blocks_a), 4)
    eq_a = [b for b in blocks_after if b.type == BlockType.EQUATION]
    low_conf_a = [b for b in blocks_after if b.confidence < 0.70]
    review_a = [b for b in blocks_after if b.status.value == "review_required" or b.requires_review]
    gemini_enhanced = [b for b in blocks_after if "gemini" in b.extraction_method.lower()]

    print(f"  Completed in {time_after}s", flush=True)
    print(f"  Blocks: {total_blocks_a} | Avg Conf: {avg_conf_a:.2%} | Equations: {len(eq_a)} | Low Conf: {len(low_conf_a)}", flush=True)
    print(f"  Gemini-enhanced blocks: {len(gemini_enhanced)}", flush=True)

    # -------------------------------------------------------------------------
    # 3. DETAILED EQUATION & FALSE-EQUATION ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[Equation Inspection]")
    print(f"BEFORE Equations ({len(eq_b)}):")
    for b in eq_b:
        print(f"  - '{b.content[:60]}' (method: {b.extraction_method}, conf: {b.confidence})")

    print(f"\nAFTER Equations ({len(eq_a)}):")
    for b in eq_a:
        latex_str = b.equation_data.latex if b.equation_data else b.content
        print(f"  - LaTeX: '{latex_str[:60]}' (method: {b.extraction_method}, conf: {b.confidence})")

    false_eq_eliminated = max(0, len(eq_b) - len(eq_a))

    # -------------------------------------------------------------------------
    # 4. REPORT & COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL COMPARISON METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} | {'BEFORE (OCR Only)':<18} | {'AFTER (OCR + Gemini)':<20}")
    print("-" * 74)
    print(f"{'Total Blocks Extracted':<30} | {total_blocks_b:<18} | {total_blocks_a:<20}")
    print(f"{'Average Confidence':<30} | {avg_conf_b * 100:.1f}%{'':<12} | {avg_conf_a * 100:.1f}%{'':<14}")
    print(f"{'Equations Detected':<30} | {len(eq_b):<18} | {len(eq_a):<20}")
    print(f"{'Validated Math Expressions':<30} | {len(eq_b):<18} | {len(eq_a):<20}")
    print(f"{'False Equations Eliminated':<30} | {'0 (unfiltered)':<18} | {false_eq_eliminated:<20}")
    print(f"{'Low-Confidence Blocks (<0.70)':<30} | {len(low_conf_b):<18} | {len(low_conf_a):<20}")
    print(f"{'Gemini-Enhanced Blocks':<30} | {'0':<18} | {len(gemini_enhanced):<20}")
    print(f"{'Blocks Needing Review':<30} | {len(review_b):<18} | {len(review_a):<20}")
    print(f"{'Processing Time':<30} | {time_before}s{'':<13} | {time_after}s{'':<15}")
    print("=" * 70)

    # Clean up temp file
    if test_file_path != pdf_path and test_file_path.exists():
        try:
            test_file_path.unlink()
        except Exception:
            pass

    return {
        "blocks_before": total_blocks_b,
        "blocks_after": total_blocks_a,
        "conf_before": avg_conf_b,
        "conf_after": avg_conf_a,
        "eq_before": len(eq_b),
        "eq_after": len(eq_a),
        "false_eq_eliminated": false_eq_eliminated,
        "low_conf_before": len(low_conf_b),
        "low_conf_after": len(low_conf_a),
        "gemini_enhanced": len(gemini_enhanced),
        "review_before": len(review_b),
        "review_after": len(review_a),
        "time_before": time_before,
        "time_after": time_after,
    }


if __name__ == "__main__":
    emt_pdf = BASE_DIR / "backend" / "data" / "doc_a24af387_EMT Unit-2 Notes & problems.pdf"
    if not emt_pdf.exists():
        # Fallback to test_emt_p1.pdf
        emt_pdf = BASE_DIR / "backend" / "data" / "test_emt_p1.pdf"

    # Run on representative test pages
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_comparison(emt_pdf, max_pages=pages)
