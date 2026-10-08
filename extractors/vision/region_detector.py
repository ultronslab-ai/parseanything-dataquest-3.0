"""Region detection using Vision LLM.
Finds structural regions in a document image for downstream targeted routing.
"""
import os
import json
from typing import List
from PIL import Image

class Region:
    def __init__(self, rtype: str, bbox: List[float], confidence: float):
        self.type = rtype  # 'text', 'handwriting', 'equation', 'figure', 'table'
        self.bbox = bbox   # [x0, y0, x1, y1] in absolute pixels
        self.confidence = confidence

class RegionDetector:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                pass

    def detect(self, image: Image.Image) -> List[Region]:
        if not self.model:
            return [Region("handwriting", [0, 0, image.width, image.height], 1.0)]

        prompt = """Analyze this document image. Identify all distinct structural regions.
Categories allowed: "text", "handwriting", "equation", "figure", "table".
Return ONLY a JSON array of objects with:
- "type": the category
- "bbox": [x_min, y_min, x_max, y_max] where coordinates are normalized from 0.0 to 1.0 (top-left is 0,0, bottom-right is 1,1).
- "confidence": 0.0 to 1.0

Do NOT include any markdown formatting, just the raw JSON array.
"""
        try:
            response = self.model.generate_content(
                [image, prompt],
                generation_config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            regions = []
            for item in data:
                t = item.get("type", "text")
                b = item.get("bbox", [0, 0, 1, 1])
                c = item.get("confidence", 0.9)
                
                # convert to pixels
                px_bbox = [
                    b[0] * image.width,
                    b[1] * image.height,
                    b[2] * image.width,
                    b[3] * image.height
                ]
                # Filter out full page boxes
                if px_bbox[2] - px_bbox[0] >= image.width * 0.95 and px_bbox[3] - px_bbox[1] >= image.height * 0.95:
                    continue
                regions.append(Region(t, px_bbox, c))
            
            # Sort regions top-to-bottom
            regions.sort(key=lambda r: r.bbox[1])
            return regions
        except Exception as e:
            print(f"[RegionDetector] Failed to detect regions: {e}")
            return [Region("handwriting", [0, 0, image.width, image.height], 1.0)]
