"""Low-Confidence Recovery Loop for Scanned and Handwritten Documents.
Evaluates OCR confidence honestly, performs adaptive preprocessing retries on low-confidence regions,
compares candidate hypotheses, preserves retry attempts, and flags persistent uncertainties without hallucination.
"""

import os
import re
import difflib
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from extractors.ocr.ocr_provider import get_ocr_provider


class ConfidenceRecoveryResult:
    def __init__(
        self,
        raw_text: str,
        final_text: str,
        confidence: float,
        requires_review: bool,
        alternatives: List[str],
        attempts: Optional[List[Dict[str, Any]]] = None,
        reason: Optional[str] = None,
        method: str = "ocr_standard",
        retry_agreement: Optional[float] = None
    ):
        self.raw_text = raw_text
        self.final_text = final_text
        self.confidence = round(max(0.0, min(1.0, float(confidence))), 4)
        self.requires_review = requires_review
        self.alternatives = [a for a in alternatives if a and a != final_text]
        self.attempts = attempts or [{"text": final_text, "confidence": self.confidence}]
        self.reason = reason
        self.method = method
        self.retry_agreement = retry_agreement

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "final_text": self.final_text,
            "confidence": self.confidence,
            "requires_review": self.requires_review,
            "alternatives": self.alternatives,
            "attempts": self.attempts,
            "ocr_attempts": len(self.attempts),
            "reason": self.reason,
            "method": self.method,
            "retry_agreement": self.retry_agreement
        }


class ConfidenceRecoveryEngine:
    """Multi-pass recovery loop for low-confidence text and handwriting regions."""

    LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.70"))
    VERY_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("VERY_LOW_CONFIDENCE_THRESHOLD", "0.45"))

    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            self._ocr = get_ocr_provider()
        return self._ocr

    def evaluate_and_recover(
        self,
        crop: Image.Image,
        initial_text: str,
        initial_confidence: float,
        region_type: str = "PARAGRAPH"
    ) -> ConfidenceRecoveryResult:
        """Evaluates confidence and executes retry loop if below threshold."""
        raw_text = initial_text.strip()
        text = raw_text
        conf = float(initial_confidence)

        # 1. Evaluate baseline validity
        printable_ratio = sum(c.isprintable() for c in text) / max(1, len(text))
        valid_ratio = sum(c.isalnum() or c.isspace() or c in "$,.()-%+:/" for c in text) / max(1, len(text))

        # Adjust initial confidence honestly based on character entropy
        if printable_ratio < 0.85 or valid_ratio < 0.50:
            conf = min(conf, 0.60)

        # If already high confidence and reasonably well-formed, accept immediately
        if conf >= self.LOW_CONFIDENCE_THRESHOLD and len(text) > 2:
            return ConfidenceRecoveryResult(
                raw_text=raw_text,
                final_text=text,
                confidence=conf,
                requires_review=False,
                alternatives=[],
                attempts=[{"text": text, "confidence": conf}],
                reason=None,
                method="ocr_accepted"
            )

        # 2. LOW CONFIDENCE TRIGGERED: Preprocess / Retry with alternate passes
        # Attempt 1: Initial pass
        attempts_log: List[Dict[str, Any]] = [{"text": text, "confidence": conf}]
        retry_candidates: List[Tuple[str, float, str]] = [(text, conf, "initial_pass")]

        try:
            # Attempt 2 (Candidate 2): CLAHE + Unsharp Masking
            retry_crop_a = self._apply_unsharp_mask(crop)
            lines_a = self._get_ocr().extract_text(retry_crop_a)
            if lines_a:
                text_a = " ".join(l.text.strip() for l in lines_a if l.text.strip())
                conf_a = sum(l.confidence for l in lines_a) / len(lines_a)
                if text_a:
                    attempts_log.append({"text": text_a, "confidence": round(conf_a, 4)})
                    retry_candidates.append((text_a, conf_a, "unsharp_mask_retry"))
        except Exception:
            pass

        try:
            # Attempt 3 (Candidate 3): Adaptive Local Otsu Binarization
            retry_crop_b = self._apply_adaptive_binarization(crop)
            lines_b = self._get_ocr().extract_text(retry_crop_b)
            if lines_b:
                text_b = " ".join(l.text.strip() for l in lines_b if l.text.strip())
                conf_b = sum(l.confidence for l in lines_b) / len(lines_b)
                if text_b:
                    attempts_log.append({"text": text_b, "confidence": round(conf_b, 4)})
                    retry_candidates.append((text_b, conf_b, "adaptive_binarization_retry"))
        except Exception:
            pass

        # 3. Compare Candidates
        valid_candidates = [c for c in retry_candidates if c[0].strip()]

        if not valid_candidates:
            return ConfidenceRecoveryResult(
                raw_text=raw_text,
                final_text=text,
                confidence=round(conf, 4),
                requires_review=True,
                alternatives=[],
                attempts=attempts_log,
                reason="Unreadable region",
                method="unrecoverable"
            )

        # Sort candidates by confidence and lexical score
        def score_candidate(cand: Tuple[str, float, str]) -> float:
            c_txt, c_conf, _ = cand
            p_ratio = sum(c.isprintable() for c in c_txt) / max(1, len(c_txt))
            has_letters = any(c.isalpha() for c in c_txt)
            letter_bonus = 0.15 if has_letters else -0.2
            return c_conf * 0.7 + p_ratio * 0.2 + letter_bonus

        valid_candidates.sort(key=score_candidate, reverse=True)
        best_text, best_conf, best_method = valid_candidates[0]

        distinct_texts = list(dict.fromkeys(c[0] for c in valid_candidates if c[0] != best_text))

        # Evaluate candidate agreement across multi-pass extraction attempts
        retry_agreement = None
        if len(valid_candidates) > 1:
            import difflib
            cand_norms = [re.sub(r"\s+", " ", c[0]).strip().lower() for c in valid_candidates]
            sims = [
                difflib.SequenceMatcher(None, cand_norms[i], cand_norms[j]).ratio()
                for i in range(len(cand_norms))
                for j in range(i + 1, len(cand_norms))
            ]
            retry_agreement = round(sum(sims) / len(sims), 4) if sims else 1.0

            # If multi-pass attempts agree strongly on the text
            if retry_agreement >= 0.85:
                best_conf = min(0.95, best_conf + (1.0 - best_conf) * 0.25 * retry_agreement)
            elif retry_agreement < 0.60:
                best_conf = min(best_conf, 0.55)

        # Check if the recovered candidate achieved reliability
        if best_conf >= self.LOW_CONFIDENCE_THRESHOLD and len(best_text) >= 3 and (retry_agreement is None or retry_agreement >= 0.65):
            return ConfidenceRecoveryResult(
                raw_text=raw_text,
                final_text=best_text,
                confidence=best_conf,
                requires_review=False,
                alternatives=distinct_texts,
                attempts=attempts_log,
                reason=None,
                method=best_method,
                retry_agreement=retry_agreement
            )

        # Still uncertain: honest flag for review with alternatives
        disagreement_reason = "Multi-pass OCR disagreement" if (retry_agreement is not None and retry_agreement < 0.65) else ("OCR disagreement" if distinct_texts else "Low extraction confidence")
        return ConfidenceRecoveryResult(
            raw_text=raw_text,
            final_text=best_text,
            confidence=round(best_conf, 4),
            requires_review=True,
            alternatives=distinct_texts,
            attempts=attempts_log,
            reason=disagreement_reason,
            method="recovery_review_flagged",
            retry_agreement=retry_agreement
        )

    def _apply_unsharp_mask(self, crop: Image.Image) -> Image.Image:
        """Sharpen faint edges using Gaussian unsharp mask."""
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        w, h = crop.size
        if h < 50:
            crop = crop.resize((int(w * 1.8), int(h * 1.8)), Image.Resampling.LANCZOS)

        img_np = np.array(crop)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.5)
        unsharp = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)
        return Image.fromarray(cv2.cvtColor(unsharp, cv2.COLOR_GRAY2RGB))

    def _apply_adaptive_binarization(self, crop: Image.Image) -> Image.Image:
        """Binarize crop adaptively for high-contrast character edges."""
        if crop.mode != "RGB":
            crop = crop.convert("RGB")
        img_np = np.array(crop)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        binarized = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
        )
        return Image.fromarray(cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB))
