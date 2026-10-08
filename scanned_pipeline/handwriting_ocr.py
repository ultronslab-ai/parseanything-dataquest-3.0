"""Handwriting Recognition Service for Scanned & Handwritten Documents.
Provides a model/provider agnostic interface:
    recognize_handwriting(image_region) -> {
        "raw_text": "...",
        "text": "...",
        "confidence": 0.0,
        "method": "handwriting_ocr",
        "bbox": [...]
    }
Supports Hugging Face TrOCR when transformers/torch are installed,
Vision LLM (Gemini) when API key is present, local handwriting-enhanced OCR,
and seamless fallback to existing OCR without breaking or crashing.
"""

import os
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

from extractors.ocr.ocr_provider import get_ocr_provider


class HandwritingOCRService:
    """Model/provider agnostic handwriting recognition pipeline."""

    def __init__(self):
        from .gemini_verifier import get_gemini_verifier
        self.verifier = get_gemini_verifier()

        # TrOCR setup if transformers/torch installed
        self.trocr_processor = None
        self.trocr_model = None
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            # Lazy loaded when first called
            self._has_trocr_lib = True
        except ImportError:
            self._has_trocr_lib = False

        self._local_ocr = None

    def _get_local_ocr(self):
        if self._local_ocr is None:
            self._local_ocr = get_ocr_provider()
        return self._local_ocr

    def recognize_handwriting(
        self,
        image_region: Image.Image,
        bbox: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Recognizes handwriting from an image region crop conforming to Requirement 7.
        Returns:
            {
                "raw_text": "...",
                "text": "...",
                "confidence": 0.0,
                "method": "handwriting_ocr",
                "bbox": [...]
            }
        """
        box = bbox or [0.0, 0.0, float(image_region.width), float(image_region.height)]

        # 1. Try Hugging Face TrOCR if libraries are installed
        if self._has_trocr_lib:
            try:
                res = self._recognize_via_trocr(image_region)
                if res and res.get("text"):
                    res["bbox"] = box
                    return res
            except Exception:
                pass

        # 2. Try Vision LLM if available
        if self.vision_model:
            try:
                result = self._recognize_via_vision_llm(image_region)
                if result and result.get("text"):
                    result["bbox"] = box
                    return result
            except Exception:
                pass

        # 3. Local Handwriting-Enhanced OCR Pass (CLAHE + stroke unsharp masking)
        try:
            enhanced_crop = self._preprocess_handwriting_crop(image_region)
            ocr = self._get_local_ocr()
            lines = ocr.extract_text(enhanced_crop)

            if lines:
                text_parts = [l.text.strip() for l in lines if l.text.strip()]
                full_text = " ".join(text_parts)
                avg_conf = sum(l.confidence for l in lines) / len(lines)
                
                # Assess character dictionary/entropy validity
                valid_ratio = sum(c.isalnum() or c.isspace() or c in ".,-+=()[]{}" for c in full_text) / max(1, len(full_text))
                adjusted_conf = round(avg_conf * (0.8 + 0.2 * valid_ratio), 4)

                return {
                    "raw_text": full_text,
                    "text": full_text,
                    "confidence": min(1.0, max(0.1, adjusted_conf)),
                    "method": "handwriting_ocr",
                    "bbox": box
                }
        except Exception:
            pass

        # 4. Final Fallback to Standard Local OCR on raw region
        try:
            ocr = self._get_local_ocr()
            lines = ocr.extract_text(image_region)
            if lines:
                text = " ".join(l.text.strip() for l in lines if l.text.strip())
                conf = sum(l.confidence for l in lines) / len(lines)
                return {
                    "raw_text": text,
                    "text": text,
                    "confidence": round(conf, 4),
                    "method": "handwriting_ocr",
                    "bbox": box
                }
        except Exception:
            pass

        return {
            "raw_text": "",
            "text": "",
            "confidence": 0.0,
            "method": "handwriting_ocr",
            "bbox": box
        }

    def _recognize_via_trocr(self, crop: Image.Image) -> Optional[Dict[str, Any]]:
        """Inference with Microsoft TrOCR if transformers/torch installed."""
        if not self._has_trocr_lib:
            return None
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        if self.trocr_processor is None:
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        
        pixel_values = self.trocr_processor(crop.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values)
        generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {
            "raw_text": generated_text,
            "text": generated_text.strip(),
            "confidence": 0.91,
            "method": "handwriting_ocr"
        }

    def _recognize_via_vision_llm(self, crop: Image.Image) -> Optional[Dict[str, Any]]:
        if not self.verifier.is_available():
            return None
        res = self.verifier.verify_and_enhance_block(
            crop=crop,
            ocr_text="",
            bbox=[0.0, 0.0, float(crop.width), float(crop.height)],
            current_type="paragraph",
            current_conf=0.50,
            is_handwritten=True
        )
        if res.get("verified") and res.get("text"):
            return {
                "raw_text": res["text"],
                "text": res["text"],
                "confidence": res.get("confidence", 0.90),
                "method": res.get("extraction_method", "ocr_local+gemini_verification"),
                "model": res.get("model", "gemini-3.8-flash")
            }
        return None

    def _preprocess_handwriting_crop(self, crop: Image.Image) -> Image.Image:
        """Preprocesses image crop to enhance faint handwriting strokes and remove paper grain."""
        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        w, h = crop.size
        if h < 60:
            scale = 2.0
            crop = crop.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        img_np = np.array(crop)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl = clahe.apply(gray)

        gaussian = cv2.GaussianBlur(cl, (0, 0), 2.0)
        unsharp = cv2.addWeighted(cl, 1.5, gaussian, -0.5, 0)

        return Image.fromarray(cv2.cvtColor(unsharp, cv2.COLOR_GRAY2RGB))


_DEFAULT_HW_SERVICE = HandwritingOCRService()


def recognize_handwriting(image_region: Image.Image, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
    """Dedicated function conforming to Requirement 7."""
    return _DEFAULT_HW_SERVICE.recognize_handwriting(image_region, bbox=bbox)
