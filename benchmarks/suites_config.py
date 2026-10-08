"""Configuration and definitions for the 13 Automated Benchmark Suites.

Defines the 13 benchmark suites, their required input files, target capabilities,
and status helpers. Strictly enforces that missing datasets display 'Dataset not available'
and unexecuted suites display 'No benchmark results available' (never 0%, failed, or fabricated).
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "benchmarks" / "benchmark_results.json"
TEST_DOCS_DIRS = [
    BASE_DIR / "test_documents",
    BASE_DIR / "benchmarks" / "data",
    BASE_DIR / "backend" / "data"
]

BENCHMARK_SUITES: List[Dict[str, Any]] = [
    {
        "id": "suite_01_digital_pdf",
        "suite_number": 1,
        "name": "Digital PDF Document",
        "format": "PDF",
        "dataset_file": "01_digital_report.pdf",
        "capability": "Vector layout, digital text extraction, heading & font hierarchy",
        "expected_behavior": "Deterministic Level 1 native text & layout extraction"
    },
    {
        "id": "suite_02_scanned_pdf",
        "suite_number": 2,
        "name": "Scanned PDF Document",
        "format": "PDF",
        "dataset_file": "02_scanned_document.pdf",
        "capability": "Raster page detection, OCR routing, confidence scoring",
        "expected_behavior": "Level 2 RapidOCR routing with word bounding boxes"
    },
    {
        "id": "suite_03_two_column_pdf",
        "suite_number": 3,
        "name": "Two-Column Layout Paper",
        "format": "PDF",
        "dataset_file": "03_two_column_paper.pdf",
        "capability": "Multi-column horizontal gutter analysis & human flow ordering",
        "expected_behavior": "Left column -> right column sequential reading order"
    },
    {
        "id": "suite_04_complex_table_pdf",
        "suite_number": 4,
        "name": "Complex Merged Table",
        "format": "PDF",
        "dataset_file": "04_complex_table.pdf",
        "capability": "Multi-row headers, merged cells (row_span & col_span)",
        "expected_behavior": "Structural 2D matrix preservation and markdown table"
    },
    {
        "id": "suite_05_multipage_table_pdf",
        "suite_number": 5,
        "name": "Multi-Page Continuous Table",
        "format": "PDF",
        "dataset_file": "05_multipage_table.pdf",
        "capability": "Cross-page table boundary alignment & table fusion",
        "expected_behavior": "Fuses Page N and N+1 matching table structures"
    },
    {
        "id": "suite_06_chart_pdf",
        "suite_number": 6,
        "name": "Figure & Chart Analysis",
        "format": "PDF",
        "dataset_file": "06_chart_document.pdf",
        "capability": "Visual figure boundary identification & anti-hallucination",
        "expected_behavior": "Preserves figure asset, avoids fabricating unverified numbers"
    },
    {
        "id": "suite_07_equation_pdf",
        "suite_number": 7,
        "name": "Mathematical Equation Document",
        "format": "PDF",
        "dataset_file": "07_equation_document.pdf",
        "capability": "Mathematical expression extraction & LaTeX normalization",
        "expected_behavior": "Outputs clean LaTeX notation ($...$ and $$...$$)"
    },
    {
        "id": "suite_08_word_docx",
        "suite_number": 8,
        "name": "Word Document (.docx)",
        "format": "DOCX",
        "dataset_file": "08_word_report.docx",
        "capability": "OpenXML heading hierarchy, bullet/numbered lists, tables",
        "expected_behavior": "Preserves native document structure and formatting"
    },
    {
        "id": "suite_09_presentation_pptx",
        "suite_number": 9,
        "name": "Presentation Deck (.pptx)",
        "format": "PPTX",
        "dataset_file": "09_presentation_deck.pptx",
        "capability": "Slide hierarchy, vector shapes, speaker notes, tables",
        "expected_behavior": "Extracts slide-by-slide content in logical order"
    },
    {
        "id": "suite_10_spreadsheet_xlsx",
        "suite_number": 10,
        "name": "Financial Spreadsheet (.xlsx)",
        "format": "XLSX",
        "dataset_file": "10_financial_model.xlsx",
        "capability": "Multi-sheet workbooks, formula evaluation, active cell ranges",
        "expected_behavior": "Extracts used ranges (A1:XN) with grid coordinates"
    },
    {
        "id": "suite_11_image_png",
        "suite_number": 11,
        "name": "Receipt / Invoice Image (.png)",
        "format": "PNG",
        "dataset_file": "11_receipt_invoice.png",
        "capability": "Raster OCR preprocessing, line & word bounding boxes",
        "expected_behavior": "Extracts text tokens with provenance coordinates"
    },
    {
        "id": "suite_12_corrupt_pdf",
        "suite_number": 12,
        "name": "Corrupted PDF File",
        "format": "PDF",
        "dataset_file": "12_corrupt_file.pdf",
        "capability": "Fast fail-safe error handling within 60s limit",
        "expected_behavior": "Non-crashing failure with CORRUPTED_FILE / PARSER_FAILURE"
    },
    {
        "id": "suite_13_unsupported_bin",
        "suite_number": 13,
        "name": "Unsupported Binary Archive (.bin)",
        "format": "BIN",
        "dataset_file": "13_unsupported_archive.bin",
        "capability": "Magic-byte inspection & fast rejection",
        "expected_behavior": "Instant rejection with UNSUPPORTED_FORMAT error code"
    }
]

def load_cached_results() -> Dict[str, Any]:
    """Loads previously persisted benchmark run results if available."""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "suites_results" in data:
                        return data["suites_results"]
                    return data
        except Exception:
            return {}
    return {}


def save_cached_results(results: Dict[str, Any]) -> None:
    """Persists benchmark run results to benchmark_results.json."""
    try:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.time() if "time" in globals() else 0,
                "suites_results": results
            }, f, indent=2)
    except Exception:
        pass


def find_dataset_file(filename: str) -> Optional[Path]:
    """Locates the dataset file across configured search directories."""
    for d in TEST_DOCS_DIRS:
        p = d / filename
        if p.exists() and p.is_file():
            return p
    return None


def get_all_suites_status(results_cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Computes transparent status for all 13 benchmark suites.
    
    If execution has not produced result data, reports 'No benchmark results available'
    rather than 0%, failed, or a fabricated score.
    If a dataset file is missing, reports 'Dataset not available'.
    """
    if results_cache is None:
        cache = load_cached_results()
    else:
        cache = results_cache

    suites_status = []
    available_datasets_count = 0
    executed_count = 0

    for suite in BENCHMARK_SUITES:
        file_path = find_dataset_file(suite["dataset_file"])
        has_file = file_path is not None
        if has_file:
            available_datasets_count += 1

        cached_res = cache.get(suite["id"])
        
        if cached_res is not None:
            executed_count += 1
            exec_status = cached_res.get("status", "Executed")
            result_display = cached_res.get("result_summary", "Completed")
            latency = cached_res.get("latency_seconds")
            blocks = cached_res.get("blocks_extracted")
            tables = cached_res.get("tables_detected")
        elif not has_file:
            exec_status = "Dataset not available"
            result_display = "No benchmark results available"
            latency = None
            blocks = None
            tables = None
        else:
            exec_status = "Ready to execute"
            result_display = "Ready to execute"
            latency = None
            blocks = None
            tables = None

        suites_status.append({
            "id": suite["id"],
            "suite_number": suite["suite_number"],
            "name": suite["name"],
            "format": suite["format"],
            "dataset_file": suite["dataset_file"],
            "dataset_available": has_file,
            "capability": suite["capability"],
            "expected_behavior": suite["expected_behavior"],
            "execution_status": exec_status,
            "result_display": result_display,
            "latency_seconds": latency,
            "blocks_extracted": blocks,
            "tables_detected": tables,
            "is_official_result": False
        })

    all_executed = (executed_count == len(BENCHMARK_SUITES))
    all_datasets_available = (available_datasets_count == len(BENCHMARK_SUITES))

    if all_executed:
        overall_display = "All 13 Suites Executed"
        status_notice = (
            "All 13 automated benchmark suites executed. "
            "Individual test files and results report are available for download."
        )
    elif executed_count > 0:
        overall_display = f"{executed_count} / {len(BENCHMARK_SUITES)} Suites Executed"
        status_notice = f"{executed_count} of 13 suites executed. Remaining suites are ready to run."
    elif all_datasets_available:
        overall_display = "Datasets Ready (13 / 13) — Click to Execute"
        status_notice = (
            "All 13 benchmark test dataset files are available and ready to run. "
            "Click 'Execute Benchmark Runner' to execute the suite or download files below."
        )
    else:
        overall_display = "No benchmark results available"
        status_notice = (
            "No official benchmark dataset or official benchmark results available. "
            "Suites without test files are marked as Dataset not available."
        )

    return {
        "title": "13x Automated Benchmark Suites",
        "has_official_results": all_executed,
        "official_dataset_available": all_datasets_available,
        "available_datasets_count": available_datasets_count,
        "total_suites_configured": len(BENCHMARK_SUITES),
        "executed_suites_count": executed_count,
        "overall_status_display": overall_display,
        "status_notice": status_notice,
        "suites": suites_status
    }

