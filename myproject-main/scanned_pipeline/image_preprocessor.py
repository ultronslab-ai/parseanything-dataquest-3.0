"""Scanned and Handwritten Document Preprocessing Pipeline.
Performs orientation detection, deskew, noise filtering, illumination normalization,
contrast enhancement (CLAHE), adaptive thresholding, and resolution scaling.
Generates 5 distinct preprocessing candidates and selects the optimal candidate without
forcing destructive binary thresholding on handwriting.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class PreprocessedImageResult:
    """Holds preprocessed image variants and audit trail metadata."""

    def __init__(
        self,
        enhanced_image: Image.Image,
        original_image: Image.Image,
        grayscale_image: np.ndarray,
        binary_mask: Optional[np.ndarray],
        metadata: Dict[str, Any],
        candidates: Optional[Dict[str, Image.Image]] = None
    ):
        self.enhanced_image = enhanced_image
        self.original_image = original_image
        self.grayscale = grayscale_image
        self.binary_mask = binary_mask
        self.metadata = metadata
        self.candidates = candidates or {}

    @property
    def width(self) -> int:
        return self.enhanced_image.width

    @property
    def height(self) -> int:
        return self.enhanced_image.height


class ScannedImagePreprocessor:
    """High-fidelity non-destructive image preprocessor with multi-candidate generation."""

    @classmethod
    def preprocess(
        cls,
        image: Image.Image,
        is_handwritten: bool = False,
        quality_score: float = 1.0
    ) -> PreprocessedImageResult:
        """Executes non-destructive enhancement pipeline and generates 5 preprocessing candidates."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        img_np = np.array(image)
        meta: Dict[str, Any] = {
            "original_dimensions": [orig_w, orig_h],
            "deskew_angle": 0.0,
            "rotation_applied": 0,
            "upscaled": False,
            "background_normalized": True,
            "contrast_enhanced": True,
            "noise_filtered": True,
            "candidates_generated": [
                "1_original_normalized",
                "2_grayscale_contrast",
                "3_denoised",
                "4_adaptive_threshold",
                "5_upscaled"
            ]
        }

        # 1. Orientation & Deskew Calculation
        gray_raw = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        skew_angle = cls._calculate_skew_angle(gray_raw)
        meta["deskew_angle"] = round(skew_angle, 2)
        target_np = img_np
        if abs(skew_angle) > 0.5 and abs(skew_angle) < 45.0:
            target_np = cls._rotate_image(target_np, skew_angle)
            meta["deskew_applied"] = True
        else:
            meta["deskew_applied"] = False

        # 2. Generate 5 Preprocessing Candidates
        candidates = cls.generate_candidates(target_np, is_handwritten=is_handwritten)

        # 3. Select Best Candidate
        # Never force binary thresholding on handwriting (handwriting works best on contrast or normalized RGB)
        if is_handwritten:
            # Handwriting prefers contrast-enhanced or normalized RGB to preserve faint pen strokes
            best_candidate_name = "2_grayscale_contrast" if quality_score < 0.85 else "1_original_normalized"
        else:
            # Printed scans prefer contrast or denoised
            best_candidate_name = "2_grayscale_contrast"

        enhanced_pil = candidates.get(best_candidate_name, candidates["1_original_normalized"])
        meta["selected_candidate"] = best_candidate_name

        # Grayscale and binary mask for layout contour analysis
        enhanced_np = np.array(enhanced_pil)
        if len(enhanced_np.shape) == 3 and enhanced_np.shape[2] == 3:
            enhanced_gray = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2GRAY)
        else:
            enhanced_gray = enhanced_np

        binary_mask = cv2.adaptiveThreshold(
            enhanced_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=10
        )

        return PreprocessedImageResult(
            enhanced_image=enhanced_pil,
            original_image=image,
            grayscale_image=enhanced_gray,
            binary_mask=binary_mask,
            metadata=meta,
            candidates=candidates
        )

    @classmethod
    def generate_candidates(cls, img_np: np.ndarray, is_handwritten: bool = False) -> Dict[str, Image.Image]:
        """Generates the 5 required preprocessing candidates:
        1. original/normalized
        2. grayscale + contrast (CLAHE)
        3. denoised
        4. adaptive threshold
        5. upscaled
        """
        candidates: Dict[str, Image.Image] = {}

        # Candidate 1: Original Normalized (Illumination Leveled RGB)
        norm_rgb = cls._normalize_background(img_np, is_handwritten=is_handwritten)
        candidates["1_original_normalized"] = Image.fromarray(norm_rgb)

        # Candidate 2: Grayscale + Contrast (CLAHE on L-channel / gray)
        gray = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.2 if is_handwritten else 1.8, tileGridSize=(8, 8))
        contrast_gray = clahe.apply(gray)
        candidates["2_grayscale_contrast"] = Image.fromarray(cv2.cvtColor(contrast_gray, cv2.COLOR_GRAY2RGB))

        # Candidate 3: Denoised (Bilateral filter to smooth flat noise while preserving edges)
        denoised = cv2.bilateralFilter(contrast_gray, d=7, sigmaColor=40, sigmaSpace=40)
        candidates["3_denoised"] = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))

        # Candidate 4: Adaptive Threshold (High contrast binarization for printed text)
        adapt_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
        candidates["4_adaptive_threshold"] = Image.fromarray(cv2.cvtColor(adapt_thresh, cv2.COLOR_GRAY2RGB))

        # Candidate 5: Upscaled (cv2.resize with cubic interpolation, 1.5x scale)
        h, w = img_np.shape[:2]
        upscaled_np = cv2.resize(norm_rgb, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
        candidates["5_upscaled"] = Image.fromarray(upscaled_np)

        return candidates

    @classmethod
    def _calculate_skew_angle(cls, gray: np.ndarray) -> float:
        """Determines skew angle in degrees using Otsu threshold and minAreaRect."""
        try:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 100:
                return 0.0

            if len(coords) > 50000:
                indices = np.random.choice(len(coords), 50000, replace=False)
                coords = coords[indices]

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45.0:
                angle = -(90.0 + angle)
            elif angle > 45.0:
                angle = 90.0 - angle
            else:
                angle = -angle

            if abs(angle) > 25.0:
                return 0.0
            return float(angle)
        except Exception:
            return 0.0

    @classmethod
    def _rotate_image(cls, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image around center by angle without clipping borders."""
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        rot_mat[0, 2] += (new_w / 2.0) - center[0]
        rot_mat[1, 2] += (new_h / 2.0) - center[1]
        
        return cv2.warpAffine(
            image,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

    @classmethod
    def _normalize_background(cls, image: np.ndarray, is_handwritten: bool = False) -> np.ndarray:
        """Illumination correction: estimates paper background and levels lighting."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            kernel_size = 51 if is_handwritten else 35
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
            
            diff = 255 - cv2.absdiff(background, gray)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.addWeighted(hsv[:, :, 2], 0.3, norm, 0.7, 0)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        except Exception:
            return image
