"""Page Quality and Document Type Classifier for ParseAnything.
Classifies individual pages into:
- DIGITAL_TEXT
- SCANNED_PRINTED
- HANDWRITTEN
- MIXED
- IMAGE_ONLY
- LOW_QUALITY_SCAN
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class PageType(str, Enum):
    DIGITAL_TEXT = "DIGITAL_TEXT"
    SCANNED_PRINTED = "SCANNED_PRINTED"
    HANDWRITTEN = "HANDWRITTEN"
    MIXED = "MIXED"
    IMAGE_ONLY = "IMAGE_ONLY"
    LOW_QUALITY_SCAN = "LOW_QUALITY_SCAN"


class PageClassificationResult:
    def __init__(
        self,
        page_type: PageType,
        confidence: float,
        metrics: Dict[str, Any],
        has_handwriting: bool = False,
        has_equations: bool = False,
        has_tables: bool = False,
        quality_score: float = 1.0,
        notes: str = ""
    ):
        self.page_type = page_type
        self.confidence = round(max(0.0, min(1.0, confidence)), 3)
        self.metrics = metrics
        self.has_handwriting = has_handwriting
        self.has_equations = has_equations
        self.has_tables = has_tables
        self.quality_score = round(max(0.0, min(1.0, quality_score)), 3)
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_type": self.page_type.value,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "has_handwriting": self.has_handwriting,
            "has_equations": self.has_equations,
            "has_tables": self.has_tables,
            "metrics": self.metrics,
            "notes": self.notes
        }


class PageQualityClassifier:
    """Classifies document page types and assesses visual quality."""

    BLUR_THRESHOLD = 40.0         # Laplacian variance below this indicates blur
    CONTRAST_THRESHOLD = 25.0     # Std dev of pixel intensity below this indicates low contrast
    MIN_DIGITAL_TEXT_CHARS = 60

    @classmethod
    def classify(
        cls,
        image: Optional[Image.Image] = None,
        native_text: str = "",
        embedded_images_count: int = 0,
        page_num: int = 1
    ) -> PageClassificationResult:
        """Classifies a page based on native text stream and visual feature analysis."""
        clean_text = (native_text or "").strip()
        cleaned_lower = clean_text.lower()
        
        # Filter scanner watermarks from native text
        for wm in ["scanned by camscanner", "camscanner", "scanned with", "created by", "document scanner"]:
            cleaned_lower = cleaned_lower.replace(wm, "")

        # 1. Check for pure Digital Text
        is_digital = len(cleaned_lower.strip()) >= cls.MIN_DIGITAL_TEXT_CHARS
        if is_digital and embedded_images_count <= 1:
            printable = sum(c.isprintable() for c in clean_text) / max(1, len(clean_text))
            if printable > 0.90:
                return PageClassificationResult(
                    page_type=PageType.DIGITAL_TEXT,
                    confidence=0.98,
                    metrics={
                        "native_char_count": len(clean_text),
                        "printable_ratio": round(printable, 3),
                        "embedded_images": embedded_images_count
                    },
                    has_handwriting=False,
                    quality_score=1.0,
                    notes="High-density native digital text stream detected."
                )

        # If no image provided, classify by available text
        if image is None:
            if is_digital:
                return PageClassificationResult(
                    page_type=PageType.DIGITAL_TEXT,
                    confidence=0.90,
                    metrics={"native_char_count": len(clean_text)},
                    quality_score=0.9
                )
            return PageClassificationResult(
                page_type=PageType.SCANNED_PRINTED,
                confidence=0.75,
                metrics={"native_char_count": len(clean_text)},
                quality_score=0.8
            )

        # 2. Visual Feature Extraction on Image
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Blur metric (Laplacian variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Contrast metric (std dev of intensities)
        contrast_std = float(np.std(gray))
        # Brightness mean
        mean_brightness = float(np.mean(gray))

        # 3. Check for LOW_QUALITY_SCAN
        is_blurred = laplacian_var < cls.BLUR_THRESHOLD
        is_low_contrast = contrast_std < cls.CONTRAST_THRESHOLD
        quality_score = min(1.0, max(0.1, (laplacian_var / 250.0) * 0.5 + (contrast_std / 75.0) * 0.5))

        if (is_blurred and is_low_contrast) or laplacian_var < 15.0 or contrast_std < 12.0:
            return PageClassificationResult(
                page_type=PageType.LOW_QUALITY_SCAN,
                confidence=0.88,
                metrics={
                    "laplacian_var": round(laplacian_var, 2),
                    "contrast_std": round(contrast_std, 2),
                    "mean_brightness": round(mean_brightness, 2),
                },
                quality_score=quality_score,
                notes="Image has severe blur or extreme low contrast."
            )

        # 4. Connected Components & Stroke Analysis for Handwriting vs Printed
        # Otsu binarization for shape inspection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(h * w)

        if edge_density < 0.005 and len(cleaned_lower.strip()) < 10:
            return PageClassificationResult(
                page_type=PageType.IMAGE_ONLY,
                confidence=0.85,
                metrics={"edge_density": round(edge_density, 4)},
                quality_score=quality_score,
                notes="Sparse edge content; classified as image/figure only."
            )

        # Contour geometry analysis
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        contour_aspect_ratios = []
        contour_heights = []
        contour_solidity = []

        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Filter noise dots and page borders
            if 6 < cw < w * 0.7 and 6 < ch < h * 0.5:
                area = cv2.contourArea(c)
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / max(1.0, hull_area)
                
                valid_contours.append((cx, cy, cw, ch))
                contour_aspect_ratios.append(float(cw) / max(1.0, float(ch)))
                contour_heights.append(ch)
                contour_solidity.append(solidity)

        handwriting_score = 0.0
        printed_score = 0.0
        has_equations = False
        has_tables = False

        if len(contour_heights) > 15:
            # Measure height variance: printed text has uniform height across characters
            std_height = float(np.std(contour_heights))
            mean_height = float(np.mean(contour_heights))
            coeff_var = std_height / max(1.0, mean_height)

            # Measure aspect ratio distribution
            mean_ar = float(np.mean(contour_aspect_ratios))
            std_ar = float(np.std(contour_aspect_ratios))

            # Handwriting typically has higher coefficient of variation in height and irregular aspect ratio
            if coeff_var > 0.70 or std_ar > 1.2:
                handwriting_score += 0.55
            else:
                printed_score += 0.55

            # Stroke regularity check using horizontal projection
            h_proj = np.sum(thresh, axis=1)
            # Peaks in horizontal projection correspond to clean printed lines
            line_peaks = [i for i in range(1, len(h_proj) - 1) if h_proj[i] > h_proj[i-1] and h_proj[i] > h_proj[i+1] and h_proj[i] > (w * 5)]
            if len(line_peaks) > 8:
                printed_score += 0.35
            else:
                handwriting_score += 0.30
        else:
            # Few contours
            printed_score += 0.3
            handwriting_score += 0.3

        # Check for table rule lines (horizontal and vertical lines)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.1), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.05)))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        if np.sum(h_lines > 0) > (w * 3) and np.sum(v_lines > 0) > (h * 2):
            has_tables = True

        # Check for mathematical symbols or formula layout
        # Mathematical formulas often have isolated symbols, fraction lines, and centered equations
        if np.sum(h_lines > 0) > (w * 0.5):
            has_equations = True

        # Combine signals
        metrics = {
            "laplacian_var": round(laplacian_var, 2),
            "contrast_std": round(contrast_std, 2),
            "edge_density": round(edge_density, 4),
            "valid_contours_count": len(valid_contours),
            "handwriting_score": round(handwriting_score, 2),
            "printed_score": round(printed_score, 2),
        }

        has_handwriting = handwriting_score > 0.45

        if is_digital and has_handwriting:
            page_type = PageType.MIXED
            conf = 0.85
        elif handwriting_score >= 0.60:
            page_type = PageType.HANDWRITTEN
            conf = min(0.95, 0.65 + handwriting_score * 0.3)
        elif printed_score >= 0.60:
            page_type = PageType.SCANNED_PRINTED
            conf = min(0.95, 0.65 + printed_score * 0.3)
        elif handwriting_score > 0.35 and printed_score > 0.35:
            page_type = PageType.MIXED
            conf = 0.80
        else:
            page_type = PageType.SCANNED_PRINTED
            conf = 0.75

        return PageClassificationResult(
            page_type=page_type,
            confidence=conf,
            metrics=metrics,
            has_handwriting=has_handwriting,
            has_equations=has_equations,
            has_tables=has_tables,
            quality_score=quality_score,
            notes=f"Classified as {page_type.value} with quality score {quality_score:.2f}."
        )
