"""Automated Benchmark Runner for ParseAnything Engine.

Checks for official benchmark dataset files, executes available suites,
and transparently reports 'Dataset not available' and 'No benchmark results available'
when official test files are not present.
Supports --smoke-test flag for local test runs (labeled: 'Local test — not an official benchmark.').
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from benchmarks.suites_config import (
    BENCHMARK_SUITES,
    find_dataset_file,
    get_all_suites_status,
    save_cached_results
)


def run_official_benchmarks() -> int:
    """Executes the 13 automated benchmark suites if dataset files exist.
    Otherwise reports 'Dataset not available' and 'No benchmark results available'.
    """
    print("=" * 70)
    print("PARSEANYTHING — 13x AUTOMATED BENCHMARK SUITES RUNNER")
    print("=" * 70)
    print("Notice: No official benchmark dataset or official results assumed.")
    print("Evaluating dataset presence for all 13 configured suites...\n")

    status = get_all_suites_status()
    results_cache = {}

    from pipeline.orchestrator import ExtractionPipeline
    pipeline = None

    for suite in status["suites"]:
        suite_name = suite["name"]
        filename = suite["dataset_file"]
        file_path = find_dataset_file(filename)

        print(f"[{suite['suite_number']:02d}/13] {suite_name} ({suite['format']}):")
        print(f"       Target File: {filename}")
        print(f"       Focus: {suite['capability']}")

        if not file_path:
            print(f"       Status: Dataset not available")
            print(f"       Result: No benchmark results available\n")
            continue

        # File is available, run real extraction!
        if pipeline is None:
            pipeline = ExtractionPipeline()

        print(f"       Input found at: {file_path}")
        start_time = time.time()
        try:
            doc = pipeline.process(file_path)
            elapsed = time.time() - start_time
            blocks_cnt = len(doc.blocks)
            tables_cnt = doc.stats.tables_detected
            status_str = "Completed" if doc.processing_status == "success" else f"Failed ({doc.errors[0].code if doc.errors else 'error'})"
            results_cache[suite["id"]] = {
                "status": "Executed",
                "result_summary": status_str,
                "latency_seconds": round(elapsed, 4),
                "blocks_extracted": blocks_cnt,
                "tables_detected": tables_cnt
            }
            print(f"       Status: Executed | Latency: {elapsed:.3f}s | Blocks: {blocks_cnt} | Status: {status_str}\n")
        except Exception as e:
            elapsed = time.time() - start_time
            results_cache[suite["id"]] = {
                "status": "Executed",
                "result_summary": f"Execution error: {e}",
                "latency_seconds": round(elapsed, 4),
                "blocks_extracted": 0
            }
            print(f"       Status: Execution error ({e})\n")

    save_cached_results(results_cache)
    final_status = get_all_suites_status(results_cache)

    print("=" * 70)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Configured Suites : {final_status['total_suites_configured']}")
    print(f"Available Datasets      : {final_status['available_datasets_count']} / {final_status['total_suites_configured']}")
    print(f"Executed Suites         : {final_status['executed_suites_count']} / {final_status['total_suites_configured']}")

    if not final_status["has_official_results"]:
        print("\nExecution Status: No benchmark results available")
        print("Note: Official benchmark dataset files are missing. Benchmark runner is")
        print("ready to execute automatically once files are added to test_documents/.\n")
    else:
        print("\nExecution Status: All 13 suites executed successfully.\n")

    return 0


def run_local_smoke_test() -> int:
    """Executes a local smoke test against bundled project documents.
    Explicitly labeled as: 'Local test — not an official benchmark.'
    """
    print("=" * 70)
    print("LOCAL SMOKE TEST — NOT AN OFFICIAL BENCHMARK")
    print("=" * 70)
    print("Running quick local integrity verification on bundled test documents.\n")

    from pipeline.orchestrator import ExtractionPipeline
    pipeline = ExtractionPipeline()

    test_targets = [
        ("Obsession_Case_Study_Answer.pdf", BASE_DIR / "Obsession_Case_Study_Answer.pdf", "Native PDF Ingestion"),
        ("Sample-Financial-Statements-image-only.pdf", BASE_DIR / "backend" / "data" / "doc_b2b733ec_Sample-Financial-Statements-image-only.pdf", "Scanned Financial OCR"),
        ("test_digital_sample.pdf", BASE_DIR / "backend" / "data" / "test_digital_sample.pdf", "Digital PDF & Table")
    ]

    for label_name, path, desc in test_targets:
        print(f"• Testing: {label_name} ({desc})")
        if not path.exists():
            print(f"  Result: File not found ({path.name})")
            continue
        try:
            t0 = time.time()
            doc = pipeline.process(path)
            t1 = time.time()
            print(f"  Result: [OK] Status: {doc.processing_status.upper()} | Time: {t1-t0:.3f}s | Blocks: {len(doc.blocks)} | Avg Conf: {round(doc.stats.average_confidence*100)}%")
        except Exception as e:
            print(f"  Result: [ERROR] {e}")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE: Local test — not an official benchmark.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ParseAnything Benchmark Runner")
    parser.add_argument("--smoke-test", action="store_true", help="Run local smoke test (not an official benchmark)")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(run_local_smoke_test())
    else:
        sys.exit(run_official_benchmarks())
