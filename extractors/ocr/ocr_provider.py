"""OCR Provider Engine for ParseAnything.
Leverages local onnxruntime RapidOCR with zero external API costs.
Provides word/line bounding boxes and confidence scores.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import numpy as np


class OCRTextLine:
    def __init__(self, text: str, bbox: List[float], confidence: float):
        self.text = text
        self.bbox = bbox  # [x_min, y_min, x_max, y_max]
        self.confidence = max(0.0, min(1.0, confidence))


class BaseOCRProvider(ABC):
    @abstractmethod
    def extract_text(self, image: Image.Image) -> List[OCRTextLine]:
        """Extract text lines with bounding boxes and confidence."""
        pass


class RapidOCRProvider(BaseOCRProvider):
    """Local, high-speed ONNX-based OCR provider."""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._engine = RapidOCR()
            except Exception as e:
                print(f"[OCRProvider] RapidOCR initialization failed: {e}")
                self._engine = None
        return self._engine

    def extract_text(self, image: Image.Image) -> List[OCRTextLine]:
        engine = self._get_engine()
        if engine is None:
            return []

        # Convert PIL image to RGB numpy array
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_np = np.array(image)

        try:
            result, _ = engine(img_np)
        except Exception as e:
            print(f"[OCRProvider] Error during OCR inference: {e}")
            return []

        if not result:
            return []

        lines: List[OCRTextLine] = []
        for item in result:
            # item format: [box_points, text, confidence]
            # box_points is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            box_points = item[0]
            text = str(item[1]).strip()
            conf = float(item[2])

            if not text:
                continue

            xs = [pt[0] for pt in box_points]
            ys = [pt[1] for pt in box_points]
            bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

            lines.append(OCRTextLine(text=text, bbox=bbox, confidence=conf))

        return lines


class FallbackOCRProvider(BaseOCRProvider):
    """Deterministic fallback provider when OCR engine is unavailable."""

    def extract_text(self, image: Image.Image) -> List[OCRTextLine]:
        return []


def get_ocr_provider() -> BaseOCRProvider:
    """Factory method to instantiate the best available local OCR provider."""
    try:
        import rapidocr_onnxruntime
        return RapidOCRProvider()
    except ImportError:
        return FallbackOCRProvider()
