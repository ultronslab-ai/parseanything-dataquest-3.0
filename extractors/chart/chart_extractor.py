"""Figure and Chart Extraction Engine for ParseAnything.
Identifies visual charts, diagrams, and figures.
Recovers structured chart data where reliable (e.g. native PPTX charts, vector diagrams),
and strictly flags partial data without hallucinating unrecoverable values.
"""

from typing import Any, Dict, List, Optional, Tuple
from schemas.models import ChartData


class ChartExtractor:
    """Extracts structured chart and figure metadata without hallucinating missing values."""

    @classmethod
    def analyze_chart_from_native(
        cls,
        title: Optional[str],
        chart_type: Optional[str],
        series_data: List[Dict[str, Any]],
        categories: Optional[List[str]] = None
    ) -> Tuple[ChartData, float]:
        """Extracts chart information when native chart objects (e.g. in PPTX / DOCX) exist."""
        labels = categories or []
        values = []
        series_names = []

        for s in series_data:
            name = s.get("name", "Series")
            series_names.append(name)
            s_vals = s.get("values", [])
            if s_vals:
                values.extend(s_vals)

        return ChartData(
            title=title or "Figure/Chart",
            chart_type=chart_type or "bar",
            labels=labels,
            values=values,
            series_names=series_names,
            status="full" if values else "partial",
            warning=None if values else "Chart values could not be reliably extracted from image"
        ), 0.95

    @classmethod
    def analyze_visual_region(
        cls,
        caption_text: Optional[str] = None,
        context_hints: Optional[List[str]] = None,
        is_vector: bool = False
    ) -> Tuple[ChartData, float]:
        """Analyzes a visual image or drawing region in PDF or image document."""
        title = caption_text
        detected_type = "diagram"

        hints = " ".join(context_hints or []).lower()
        if caption_text:
            hints += " " + caption_text.lower()

        if any(w in hints for w in ["bar chart", "bar graph", "revenue growth", "distribution"]):
            detected_type = "bar"
        elif any(w in hints for w in ["line chart", "trend", "timeline", "over time"]):
            detected_type = "line"
        elif any(w in hints for w in ["pie chart", "share", "percentage breakdown"]):
            detected_type = "pie"
        elif any(w in hints for w in ["chart", "plot", "graph"]):
            detected_type = "chart"

        # Under the strict rule: "If exact numeric data cannot be reliably recovered: DO NOT INVENT IT."
        return ChartData(
            title=title or "Visual Chart/Figure",
            chart_type=detected_type,
            labels=[],
            values=[],
            series_names=[],
            status="partial",
            warning="Chart values could not be reliably extracted without external vision model"
        ), 0.85
