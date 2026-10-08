"""Deterministic Generator for 13 Benchmark Test Documents using PyMuPDF, DOCX, PPTX, XLSX, and PIL.

Generates genuine, high-fidelity test files corresponding to the 13 automated
benchmark suites for ParseAnything engine evaluation.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import pymupdf
import docx
import pptx
from pptx.util import Inches as PPTInches
import openpyxl

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "test_documents"


def generate_01_digital_report(out_dir: Path):
    """01_digital_report.pdf: Clean digital PDF with heading hierarchy and table."""
    pdf_path = out_dir / "01_digital_report.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    # Document Header
    page.insert_text(pymupdf.Point(54, 70), "Global Tech Operations Annual Report", fontsize=18, fontname="hebo", color=(0.1, 0.15, 0.25))
    page.insert_text(pymupdf.Point(54, 95), "Executive Summary & Core Operational Metrics", fontsize=13, fontname="hebo", color=(0.2, 0.3, 0.45))
    
    body = (
        "This digital document demonstrates native vector layout extraction, preserved typography, "
        "and semantic block classification. All fonts and character geometries are preserved."
    )
    page.insert_textbox(pymupdf.Rect(54, 110, 558, 145), body, fontsize=10, fontname="helv", color=(0.2, 0.25, 0.35))

    page.insert_text(pymupdf.Point(54, 170), "Regional Performance Metrics (FY2026)", fontsize=12, fontname="hebo", color=(0.1, 0.15, 0.25))

    # Draw Table
    headers = ["Region", "Target ($M)", "Actual ($M)", "Growth (%)"]
    rows = [
        ["North America", "250.0", "284.5", "+13.8%"],
        ["Europe & UK", "180.0", "195.2", "+8.4%"],
        ["Asia Pacific", "220.0", "248.9", "+13.1%"],
        ["Latin America", "75.0", "82.4", "+9.8%"]
    ]
    col_x = [54, 190, 310, 430, 550]
    y = 185
    # Header row
    page.draw_rect(pymupdf.Rect(54, y, 550, y + 24), color=(0.15, 0.2, 0.3), fill=(0.15, 0.2, 0.3))
    for c_idx, h in enumerate(headers):
        page.insert_text(pymupdf.Point(col_x[c_idx] + 8, y + 16), h, fontsize=10, fontname="hebo", color=(1, 1, 1))

    y += 24
    for r_idx, r in enumerate(rows):
        bg = (0.96, 0.97, 0.98) if r_idx % 2 == 1 else (1, 1, 1)
        page.draw_rect(pymupdf.Rect(54, y, 550, y + 22), color=(0.85, 0.88, 0.92), fill=bg)
        for c_idx, val in enumerate(r):
            page.insert_text(pymupdf.Point(col_x[c_idx] + 8, y + 15), val, fontsize=9.5, fontname="helv", color=(0.2, 0.25, 0.35))
        y += 22

    doc.save(str(pdf_path))
    doc.close()


def generate_02_scanned_document(out_dir: Path):
    """02_scanned_document.pdf: Scanned image-only PDF requiring OCR."""
    pdf_path = out_dir / "02_scanned_document.pdf"
    img = Image.new("RGB", (1000, 1400), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)

    draw.text((80, 100), "CLINICAL AUDIT & MEDICAL EVALUATION", fill=(30, 41, 59))
    draw.text((80, 150), "Patient Record ID: MED-2026-88914", fill=(71, 85, 105))
    draw.text((80, 200), "Diagnosis Summary: Acute Respiratory Evaluation", fill=(51, 65, 85))
    draw.text((80, 250), "Prescription: Amoxicillin 500mg Oral Twice Daily", fill=(51, 65, 85))
    draw.text((80, 300), "Follow-up Scheduled: October 24, 2026", fill=(51, 65, 85))
    draw.text((80, 370), "Physician Signature Verified: Dr. A. Vance, MD", fill=(30, 41, 59))

    tmp_png = out_dir / "temp_scanned_02.png"
    img.save(str(tmp_png), format="PNG")

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(tmp_png))
    doc.save(str(pdf_path))
    doc.close()
    if tmp_png.exists():
        tmp_png.unlink()


def generate_03_two_column_paper(out_dir: Path):
    """03_two_column_paper.pdf: Academic paper with two-column layout."""
    pdf_path = out_dir / "03_two_column_paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text(pymupdf.Point(54, 60), "Deep Learning Approaches to Universal Document Parsing", fontsize=16, fontname="hebo", color=(0.1, 0.15, 0.25))
    page.insert_text(pymupdf.Point(54, 85), "Research Team | Published October 2026", fontsize=10, fontname="helv", color=(0.4, 0.45, 0.55))

    left_lines = [
        "1. Introduction",
        "Document ingestion platforms face challenges",
        "when multi-column typography splits visual sentences.",
        "Traditional horizontal text sorting merges lines",
        "from adjacent columns incorrectly into single phrases.",
        "",
        "2. Related Work",
        "Heuristic layout models detect vertical gutters",
        "to segment left and right text streams sequentially.",
        "Our approach preserves reading order cleanly."
    ]
    y = 120
    for line in left_lines:
        is_heading = line.startswith("1.") or line.startswith("2.")
        size = 11.5 if is_heading else 9.5
        font = "hebo" if is_heading else "helv"
        color = (0.05, 0.1, 0.2) if is_heading else (0.2, 0.25, 0.35)
        page.insert_text(pymupdf.Point(54, y), line, fontsize=size, fontname=font, color=color)
        y += 18 if is_heading else 14

    right_lines = [
        "3. Experimental Methodology",
        "We evaluate candidate document extractors",
        "against 13 multi-format suites covering tables,",
        "equations, scans, and presentation shapes.",
        "",
        "4. Quantitative Results",
        "The cascading architecture achieves 100% flow",
        "integrity across multi-column test runs.",
        "Throughput exceeds 10 pages per second.",
        "",
        "5. Conclusion",
        "Gutter-aware parsing guarantees fidelity."
    ]
    y = 120
    for line in right_lines:
        is_heading = line.startswith("3.") or line.startswith("4.") or line.startswith("5.")
        size = 11.5 if is_heading else 9.5
        font = "hebo" if is_heading else "helv"
        color = (0.05, 0.1, 0.2) if is_heading else (0.2, 0.25, 0.35)
        page.insert_text(pymupdf.Point(320, y), line, fontsize=size, fontname=font, color=color)
        y += 18 if is_heading else 14

    doc.save(str(pdf_path))
    doc.close()


def generate_04_complex_table(out_dir: Path):
    """04_complex_table.pdf: Complex table with merged cells and headers."""
    pdf_path = out_dir / "04_complex_table.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text(pymupdf.Point(54, 70), "Consolidated Balance Matrix", fontsize=18, fontname="hebo", color=(0.1, 0.15, 0.25))
    page.insert_text(pymupdf.Point(54, 95), "Complex multi-row header and merged cell matrix validation.", fontsize=10, fontname="helv", color=(0.4, 0.45, 0.55))

    # Draw table matrix
    y = 130
    page.draw_rect(pymupdf.Rect(54, y, 550, y + 26), color=(0.1, 0.15, 0.25), fill=(0.1, 0.15, 0.25))
    page.insert_text(pymupdf.Point(62, y + 17), "Division", fontsize=10, fontname="hebo", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(210, y + 17), "Fiscal Year 2026", fontsize=10, fontname="hebo", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(390, y + 17), "Fiscal Year 2025", fontsize=10, fontname="hebo", color=(1, 1, 1))

    y += 26
    page.draw_rect(pymupdf.Rect(54, y, 550, y + 22), color=(0.2, 0.25, 0.35), fill=(0.2, 0.25, 0.35))
    page.insert_text(pymupdf.Point(210, y + 15), "Q1 Actual", fontsize=9, fontname="hebo", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(300, y + 15), "Q2 Budget", fontsize=9, fontname="hebo", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(390, y + 15), "Q1 Actual", fontsize=9, fontname="hebo", color=(1, 1, 1))
    page.insert_text(pymupdf.Point(480, y + 15), "Q2 Budget", fontsize=9, fontname="hebo", color=(1, 1, 1))

    rows = [
        ["Enterprise Cloud", "$45.2M", "$48.0M", "$38.1M", "$40.5M"],
        ["Consumer Hardware", "$32.8M", "$35.0M", "$31.4M", "$33.0M"],
        ["Professional Services", "$18.5M", "$20.0M", "$16.2M", "$17.5M"],
        ["Total Revenue", "$96.5M", "$103.0M", "$85.7M", "$91.0M"]
    ]
    col_x = [54, 200, 290, 380, 470]
    y += 22
    for r in rows:
        page.draw_rect(pymupdf.Rect(54, y, 550, y + 20), color=(0.85, 0.88, 0.92))
        for idx, val in enumerate(r):
            page.insert_text(pymupdf.Point(col_x[idx] + 8, y + 14), val, fontsize=9, fontname="helv", color=(0.15, 0.2, 0.3))
        y += 20

    doc.save(str(pdf_path))
    doc.close()


def generate_05_multipage_table(out_dir: Path):
    """05_multipage_table.pdf: Table spanning across pages 1 and 2 for cross-page merge."""
    pdf_path = out_dir / "05_multipage_table.pdf"
    doc = pymupdf.open()
    
    headers = ["Account Code", "Account Description", "Debit ($)", "Credit ($)", "Status"]
    col_x = [54, 150, 350, 440, 520]

    # Page 1
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(pymupdf.Point(54, 60), "Multi-Page Continuous General Ledger (Page 1)", fontsize=16, fontname="hebo", color=(0.1, 0.15, 0.25))
    y = 90
    p1.draw_rect(pymupdf.Rect(54, y, 558, y + 24), color=(0.15, 0.2, 0.3), fill=(0.15, 0.2, 0.3))
    for c_idx, h in enumerate(headers):
        p1.insert_text(pymupdf.Point(col_x[c_idx] + 4, y + 16), h, fontsize=9.5, fontname="hebo", color=(1, 1, 1))

    y += 24
    for i in range(1, 26):
        p1.draw_rect(pymupdf.Rect(54, y, 558, y + 22), color=(0.85, 0.88, 0.92))
        row = [f"GL-{1000 + i}", f"Operating Expense Line Item #{i}", f"{i * 115.50:.2f}", "0.00", "Posted"]
        for c_idx, val in enumerate(row):
            p1.insert_text(pymupdf.Point(col_x[c_idx] + 4, y + 15), val, fontsize=9, fontname="helv", color=(0.2, 0.25, 0.35))
        y += 22

    # Page 2 (Continuing table)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(pymupdf.Point(54, 60), "Multi-Page Continuous General Ledger (Page 2)", fontsize=16, fontname="hebo", color=(0.1, 0.15, 0.25))
    y = 90
    p2.draw_rect(pymupdf.Rect(54, y, 558, y + 24), color=(0.15, 0.2, 0.3), fill=(0.15, 0.2, 0.3))
    for c_idx, h in enumerate(headers):
        p2.insert_text(pymupdf.Point(col_x[c_idx] + 4, y + 16), h, fontsize=9.5, fontname="hebo", color=(1, 1, 1))

    y += 24
    for i in range(26, 48):
        p2.draw_rect(pymupdf.Rect(54, y, 558, y + 22), color=(0.85, 0.88, 0.92))
        row = [f"GL-{1000 + i}", f"Operating Expense Line Item #{i}", f"{i * 115.50:.2f}", "0.00", "Posted"]
        for c_idx, val in enumerate(row):
            p2.insert_text(pymupdf.Point(col_x[c_idx] + 4, y + 15), val, fontsize=9, fontname="helv", color=(0.2, 0.25, 0.35))
        y += 22

    doc.save(str(pdf_path))
    doc.close()


def generate_06_chart_document(out_dir: Path):
    """06_chart_document.pdf: PDF containing a visual chart plot."""
    pdf_path = out_dir / "06_chart_document.pdf"
    
    # Draw chart with PIL
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20, 20), (580, 280)], outline=(226, 232, 240), width=1)
    draw.text((150, 40), "Annual Scaled Throughput Progression", fill=(15, 23, 42))

    bars = [("Q1", 50, 80), ("Q2", 170, 130), ("Q3", 290, 180), ("Q4", 410, 230)]
    for label, x, h in bars:
        draw.rectangle([(x, 250 - h), (x + 60, 250)], fill=(14, 165, 233))
        draw.text((x + 18, 255), label, fill=(71, 85, 105))

    tmp_png = out_dir / "temp_chart_06.png"
    img.save(str(tmp_png), format="PNG")

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(pymupdf.Point(54, 70), "Visual Analytics & Architectural Performance", fontsize=18, fontname="hebo", color=(0.1, 0.15, 0.25))
    page.insert_text(pymupdf.Point(54, 95), "Document containing visual figure to test chart and figure boundary extraction.", fontsize=10, fontname="helv", color=(0.3, 0.35, 0.45))
    page.insert_image(pymupdf.Rect(54, 130, 558, 380), filename=str(tmp_png))
    page.insert_text(pymupdf.Point(54, 405), "Figure 1: Measured throughput scaling across automated benchmark execution.", fontsize=9.5, fontname="hebo", color=(0.4, 0.45, 0.55))

    doc.save(str(pdf_path))
    doc.close()
    if tmp_png.exists():
        tmp_png.unlink()


def generate_07_equation_document(out_dir: Path):
    """07_equation_document.pdf: Document with mathematical equations and LaTeX expressions."""
    pdf_path = out_dir / "07_equation_document.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    page.insert_text(pymupdf.Point(54, 70), "Theoretical Physics & Mathematical Equations", fontsize=18, fontname="hebo", color=(0.1, 0.15, 0.25))
    page.insert_text(pymupdf.Point(54, 95), "Mathematical expressions and LaTeX formulas validation.", fontsize=10, fontname="helv", color=(0.3, 0.35, 0.45))

    eqs = [
        ("1. Mass-Energy Equivalence Formula:", "$$E = m c^2$$"),
        ("2. Gaussian Normal Distribution Density:", "$$f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}$$"),
        ("3. Standard Deviation Summation:", "$$\\sigma = \\sqrt{\\frac{1}{N}\\sum_{i=1}^N (x_i - \\mu)^2}$$")
    ]
    y = 140
    for title, formula in eqs:
        page.insert_text(pymupdf.Point(54, y), title, fontsize=12, fontname="hebo", color=(0.1, 0.15, 0.25))
        y += 24
        page.draw_rect(pymupdf.Rect(54, y - 14, 558, y + 18), color=(0.85, 0.9, 0.95), fill=(0.95, 0.98, 1.0))
        page.insert_text(pymupdf.Point(70, y + 6), formula, fontsize=11, fontname="courier", color=(0.02, 0.45, 0.7))
        y += 45

    doc.save(str(pdf_path))
    doc.close()


def generate_08_word_report(out_dir: Path):
    """08_word_report.docx: Word document with Heading hierarchy, bullet lists, and tables."""
    docx_path = out_dir / "08_word_report.docx"
    doc = docx.Document()
    doc.add_heading("Enterprise Compliance and Governance Framework", level=1)
    doc.add_paragraph("This Microsoft Word document validates OpenXML DOM traversal, heading hierarchy, and nested table extraction.")
    
    doc.add_heading("1. Strategic Security Mandates", level=2)
    doc.add_paragraph("Mandatory encryption at rest using AES-256 standards.", style='List Bullet')
    doc.add_paragraph("Automated role-based access control with biometric verification.", style='List Bullet')
    doc.add_paragraph("Quarterly third-party vulnerability assessments.", style='List Bullet')

    doc.add_heading("2. Audit Assessment Matrix", level=2)
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Control ID"
    hdr_cells[1].text = "Requirement"
    hdr_cells[2].text = "Frequency"
    hdr_cells[3].text = "Compliance Status"

    sample_rows = [
        ("SEC-01", "Multi-Factor Authentication", "Daily / Continuous", "Compliant"),
        ("SEC-02", "Database Key Rotation", "90 Days", "Compliant"),
        ("SEC-03", "Immutable Audit Logging", "Real-Time", "Compliant")
    ]
    for cid, req, freq, stat in sample_rows:
        row_cells = table.add_row().cells
        row_cells[0].text = cid
        row_cells[1].text = req
        row_cells[2].text = freq
        row_cells[3].text = stat

    doc.save(str(docx_path))


def generate_09_presentation_deck(out_dir: Path):
    """09_presentation_deck.pptx: PowerPoint presentation with slides, shapes, and tables."""
    pptx_path = out_dir / "09_presentation_deck.pptx"
    prs = pptx.Presentation()

    # Slide 1: Title slide
    title_layout = prs.slide_layouts[0]
    slide1 = prs.slides.add_slide(title_layout)
    slide1.shapes.title.text = "ParseAnything 2026 Product Strategy"
    slide1.placeholders[1].text = "Autonomous Ingestion, Provenance Inspection, and Multimodal Scale"

    # Slide 2: Table slide
    blank_layout = prs.slide_layouts[5]
    slide2 = prs.slides.add_slide(blank_layout)
    slide2.shapes.title.text = "Competitive Benchmark Comparison"
    
    rows, cols = 4, 3
    left, top, width, height = PPTInches(1), PPTInches(2), PPTInches(8), PPTInches(3)
    table_shape = slide2.shapes.add_table(rows, cols, left, top, width, height)
    tbl = table_shape.table
    tbl.cell(0, 0).text = "Platform Engine"
    tbl.cell(0, 1).text = "BBox Provenance"
    tbl.cell(0, 2).text = "Cost per 1k Pages"
    
    data = [
        ("ParseAnything Universal", "Yes (Pixel Exact)", "$0.00 - $0.50"),
        ("Traditional OCR", "Partial Coordinates", "$1.50 - $3.00"),
        ("Cloud Vision LLM", "No Provenance", "$30.00 - $60.00")
    ]
    for r_idx, (p, b, c) in enumerate(data, 1):
        tbl.cell(r_idx, 0).text = p
        tbl.cell(r_idx, 1).text = b
        tbl.cell(r_idx, 2).text = c

    prs.save(str(pptx_path))


def generate_10_financial_model(out_dir: Path):
    """10_financial_model.xlsx: Multi-sheet Excel workbook with cell ranges and formula values."""
    xlsx_path = out_dir / "10_financial_model.xlsx"
    wb = openpyxl.Workbook()
    
    ws1 = wb.active
    ws1.title = "Income Forecast"
    ws1.append(["Account Category", "Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026", "Total FY26"])
    ws1.append(["Software Revenue", 1250000, 1420000, 1680000, 1950000, "=SUM(B2:E2)"])
    ws1.append(["Professional Services", 320000, 340000, 390000, 420000, "=SUM(B3:E3)"])
    ws1.append(["Cost of Goods Sold", 280000, 310000, 340000, 380000, "=SUM(B4:E4)"])
    ws1.append(["Gross Profit", "=B2+B3-B4", "=C2+C3-C4", "=D2+D3-D4", "=E2+E3-E4", "=F2+F3-F4"])

    ws2 = wb.create_sheet(title="Balance Sheet")
    ws2.append(["Assets Line Item", "Book Value ($)", "Current Ratio"])
    ws2.append(["Cash & Equivalents", 4500000, 1.85])
    ws2.append(["Accounts Receivable", 1250000, 1.20])
    ws2.append(["Property & Equipment", 8400000, 0.95])
    ws2.append(["Total Assets", 14150000, "Healthy"])

    wb.save(str(xlsx_path))


def generate_11_receipt_invoice(out_dir: Path):
    """11_receipt_invoice.png: High-resolution raster receipt image."""
    png_path = out_dir / "11_receipt_invoice.png"
    img = Image.new("RGB", (800, 1100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((250, 50), "METRO LOGISTICS INC.", fill=(15, 23, 42))
    draw.text((230, 80), "Official Commercial Invoice #90214", fill=(100, 116, 139))
    draw.line([(50, 120), (750, 120)], fill=(203, 213, 225), width=2)

    draw.text((60, 150), "Item Description", fill=(15, 23, 42))
    draw.text((450, 150), "Qty", fill=(15, 23, 42))
    draw.text((550, 150), "Unit Price", fill=(15, 23, 42))
    draw.text((670, 150), "Total", fill=(15, 23, 42))
    draw.line([(50, 180), (750, 180)], fill=(203, 213, 225), width=1)

    items = [
        ("High-Speed Fiber Terminal Router", "2", "$450.00", "$900.00"),
        ("Server Rack Mount Rails (2U)", "4", "$85.00", "$340.00"),
        ("Cat6A Shielded Patch Cable 50ft", "10", "$18.50", "$185.00"),
        ("On-Site Professional Installation", "1", "$650.00", "$650.00")
    ]
    y = 210
    for desc, qty, unit, total in items:
        draw.text((60, y), desc, fill=(51, 65, 85))
        draw.text((460, y), qty, fill=(51, 65, 85))
        draw.text((550, y), unit, fill=(51, 65, 85))
        draw.text((670, y), total, fill=(51, 65, 85))
        y += 45

    draw.line([(50, y + 20), (750, y + 20)], fill=(203, 213, 225), width=2)
    draw.text((520, y + 50), "Subtotal:", fill=(51, 65, 85))
    draw.text((670, y + 50), "$2,075.00", fill=(51, 65, 85))
    draw.text((520, y + 90), "Tax (8.25%):", fill=(51, 65, 85))
    draw.text((670, y + 90), "$171.19", fill=(51, 65, 85))
    draw.text((520, y + 130), "Final Balance:", fill=(15, 23, 42))
    draw.text((660, y + 130), "$2,246.19", fill=(15, 23, 42))

    img.save(str(png_path), format="PNG")


def generate_12_corrupt_file(out_dir: Path):
    """12_corrupt_file.pdf: Corrupted PDF file to validate fail-safe handling."""
    pdf_path = out_dir / "12_corrupt_file.pdf"
    corrupt_bytes = (
        b"%PDF-1.4\n"
        b"%CORRUPTED_STREAM_HEADER\n"
        b"\x00\xff\xfe\x01\x02\x03\x04INVALID_TRUNCATED_OBJECT_STREAM\n"
        b"trailer\n<< /Size 1 >>\nstartxref\n999999\n%%EOF\n"
    )
    with open(pdf_path, "wb") as f:
        f.write(corrupt_bytes)


def generate_13_unsupported_archive(out_dir: Path):
    """13_unsupported_archive.bin: Unsupported binary format for fast rejection."""
    bin_path = out_dir / "13_unsupported_archive.bin"
    unsupported_bytes = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00>\x00\x01\x00\x00\x00UNSUPPORTED_BINARY"
    with open(bin_path, "wb") as f:
        f.write(unsupported_bytes)


def generate_all_benchmark_documents(out_dir: Path = None):
    """Generates all 13 genuine benchmark test files."""
    dest = out_dir or OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Generating 13 benchmark test documents into: {dest}...")

    generate_01_digital_report(dest)
    print("  [OK] 01_digital_report.pdf")
    generate_02_scanned_document(dest)
    print("  [OK] 02_scanned_document.pdf")
    generate_03_two_column_paper(dest)
    print("  [OK] 03_two_column_paper.pdf")
    generate_04_complex_table(dest)
    print("  [OK] 04_complex_table.pdf")
    generate_05_multipage_table(dest)
    print("  [OK] 05_multipage_table.pdf")
    generate_06_chart_document(dest)
    print("  [OK] 06_chart_document.pdf")
    generate_07_equation_document(dest)
    print("  [OK] 07_equation_document.pdf")
    generate_08_word_report(dest)
    print("  [OK] 08_word_report.docx")
    generate_09_presentation_deck(dest)
    print("  [OK] 09_presentation_deck.pptx")
    generate_10_financial_model(dest)
    print("  [OK] 10_financial_model.xlsx")
    generate_11_receipt_invoice(dest)
    print("  [OK] 11_receipt_invoice.png")
    generate_12_corrupt_file(dest)
    print("  [OK] 12_corrupt_file.pdf")
    generate_13_unsupported_archive(dest)
    print("  [OK] 13_unsupported_archive.bin")
    print("All 13 benchmark documents generated successfully!\n")


if __name__ == "__main__":
    generate_all_benchmark_documents()
