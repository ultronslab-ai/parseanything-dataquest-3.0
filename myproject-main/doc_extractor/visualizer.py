"""Document Visualizer.

Renders document pages with color-coded bounding boxes, element type tags,
and confidence scores overlayed on the original page image.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from doc_extractor.models import ElementType, ExtractedElement, PageExtractionResult

# Color schemes by element type (RGB tuples)
TYPE_COLORS: Dict[str, Tuple[int, int, int]] = {
    ElementType.TEXT.value: (59, 130, 246),      # Blue
    ElementType.TABLE.value: (16, 185, 129),     # Green
    ElementType.CHART.value: (245, 158, 11),     # Amber / Orange
    ElementType.EQUATION.value: (139, 92, 246),  # Purple
}


class DocumentVisualizer:
    """Draws bounding boxes and labels onto document pages."""

    def __init__(self, line_width: int = 3, show_labels: bool = True):
        self.line_width = line_width
        self.show_labels = show_labels
        self._font = self._load_default_font()

    def _load_default_font(self):
        try:
            # Try to load Arial or similar TTF if available
            return ImageFont.truetype("arial.ttf", 14)
        except Exception:
            return ImageFont.load_default()

    def annotate_page(
        self,
        base_image: Image.Image,
        elements: List[ExtractedElement],
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> Image.Image:
        """Draws bounding boxes and labels for all elements on the base image."""
        img_copy = base_image.copy().convert("RGB")
        draw = ImageDraw.Draw(img_copy)

        for elem in elements:
            color = TYPE_COLORS.get(elem.type.value, (120, 120, 120))
            bbox = elem.bbox

            # Adjust if scale factor is needed (e.g. if bbox is in 72 DPI PDF points vs 150 DPI pixel image)
            x0 = bbox.x_min * scale_x
            y0 = bbox.y_min * scale_y
            x1 = bbox.x_max * scale_x
            y1 = bbox.y_max * scale_y

            # Draw outer rectangle
            draw.rectangle([x0, y0, x1, y1], outline=color, width=self.line_width)

            if self.show_labels:
                label_text = f" {elem.type.value.upper()} | {elem.confidence:.2f} "
                
                # Calculate text size for badge background
                try:
                    left, top, right, bottom = self._font.getbbox(label_text)
                    text_w = right - left
                    text_h = bottom - top
                except Exception:
                    text_w = len(label_text) * 8
                    text_h = 16

                badge_y0 = max(0, y0 - text_h - 4)
                badge_y1 = badge_y0 + text_h + 4
                badge_x1 = x0 + text_w + 6

                # Draw badge background
                draw.rectangle([x0, badge_y0, badge_x1, badge_y1], fill=color)
                # Draw badge text (white)
                draw.text((x0 + 3, badge_y0 + 2), label_text, fill=(255, 255, 255), font=self._font)

        return img_copy

    def render_and_annotate_pdf(
        self,
        pdf_path: Union[str, Path],
        page_results: List[PageExtractionResult],
        output_dir: Union[str, Path],
        dpi: int = 150
    ) -> List[str]:
        """Renders all PDF pages to disk with annotations."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []

        doc = pymupdf.open(str(pdf_path))

        for page_res in page_results:
            page_idx = page_res.page_number - 1
            if page_idx >= len(doc):
                continue

            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Calculate scale factor between PDF points (72 DPI) and rendered pixmap
            scale_x = pix.width / page.rect.width
            scale_y = pix.height / page.rect.height

            annotated_img = self.annotate_page(img, page_res.elements, scale_x, scale_y)

            filename = f"page_{page_res.page_number}_annotated.png"
            dest_path = out_dir / filename
            annotated_img.save(dest_path)
            saved_paths.append(str(dest_path))

        doc.close()
        return saved_paths
