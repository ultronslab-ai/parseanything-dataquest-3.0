"""Command Line Interface for Document Extraction Pipeline."""

import argparse
import sys
from pathlib import Path
from doc_extractor.pipeline import DocumentExtractionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured elements (text, table, chart, equation) from documents."
    )
    parser.add_argument("input", type=str, help="Path to input PDF or image file")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Directory to save extracted JSON and annotated visualizations"
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["hybrid", "gemini"],
        default="hybrid",
        help="Extraction engine: 'hybrid' (offline local PyMuPDF) or 'gemini' (Vision LLM)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering DPI for visualization"
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable generating annotated visual bounding box images"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: File '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    # Ensure UTF-8 output on Windows console
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 55)
    print(f"[*] Processing: {input_file.name}")
    print(f"[*] Engine:     {args.engine.upper()}")
    print(f"[*] Output Dir: {args.output}")
    print("=" * 55 + "\n")

    pipeline = DocumentExtractionPipeline(
        engine=args.engine,
        render_dpi=args.dpi,
        enable_visualization=not args.no_vis
    )

    try:
        result = pipeline.process(file_path=input_file, output_dir=args.output)
        
        print("\n[+] Extraction Successful!")
        print(f"  - Total Elements Extracted: {len(result.elements)}")
        print(f"  - Average Confidence:        {result.average_confidence:.1%}")
        print(f"  - Processing Time:           {result.processing_time_seconds:.2f}s")
        print("\nBreakdown by Type:")
        for elem_type, count in result.counts_by_type.items():
            print(f"  - {elem_type.capitalize()}: {count}")

        print(f"\n[+] Results saved to: {Path(args.output).resolve()}")
    except Exception as e:
        print(f"\n[-] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
