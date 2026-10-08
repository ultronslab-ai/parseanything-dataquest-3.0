"""Sample PDF Generator with Text, Table, Chart, and Equation.

Creates a multi-element document to validate extraction pipelines.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage


def generate_sample_pdf(output_path: str = "sample_document.pdf") -> str:
    dest = Path(output_path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate a sample chart image using matplotlib
    chart_img_path = dest.parent / "temp_chart.png"
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=150)
    quarters = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]
    revenue = [12.4, 14.8, 18.2, 22.1]
    ax.bar(quarters, revenue, color="#2563eb", width=0.5)
    ax.set_title("Quarterly Revenue Growth (in $B)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue ($B)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(chart_img_path, format="png")
    plt.close(fig)

    # 2. Build PDF document using ReportLab
    doc = SimpleDocTemplate(
        str(dest),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=12
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10
    )
    math_style = ParagraphStyle(
        'DocMath',
        parent=styles['Normal'],
        fontSize=13,
        leading=18,
        alignment=1,  # Centered
        textColor=colors.HexColor("#6b21a8"),
        spaceBefore=10,
        spaceAfter=14
    )

    story = []

    # Title & Text
    story.append(Paragraph("Quarterly Financial & Analytical Report", title_style))
    story.append(Paragraph(
        "This report provides an empirical analysis of operational throughput, regional performance, and risk metrics. "
        "All measurements were computed using standardized fiscal variance equations.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Table
    table_data = [
        ["Region", "Target ($M)", "Actual ($M)", "Variance (%)"],
        ["North America", "120.0", "135.4", "+12.8%"],
        ["Europe", "95.0", "98.2", "+3.3%"],
        ["Asia-Pacific", "110.0", "122.5", "+11.3%"],
        ["Latin America", "45.0", "42.1", "-6.4%"]
    ]
    t = Table(table_data, colWidths=[130, 110, 110, 110])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Equation
    story.append(Paragraph("Variance and Volatility Metric Formulation:", body_style))
    story.append(Paragraph("σ = √( 1/N * ∑ (x_i - μ)^2 )", math_style))
    story.append(Paragraph("E = mc^2", math_style))
    story.append(Spacer(1, 10))

    # Chart
    story.append(RLImage(str(chart_img_path), width=480, height=200))

    # Build PDF
    doc.build(story)

    # Cleanup temp chart image
    if chart_img_path.exists():
        chart_img_path.unlink()

    print(f"Sample PDF successfully generated at: {dest}")
    return str(dest)


if __name__ == "__main__":
    generate_sample_pdf()
