"""Cost and Performance Instrumentation Manager for ParseAnything.
Tracks token usage, external API calls, local processing throughput,
and calculates estimated cost per 1,000 pages against the <$10/1k pages DQCL threshold.
"""

import os
from typing import Dict, Any


class CostManager:
    """Tracks model costs and verifies budget targets."""

    def __init__(self):
        # Configurable API pricing (per 1,000 requests/pages or per million tokens)
        # Default local deterministic/OCR has $0.00 external API cost!
        self.vision_api_cost_per_page = float(os.getenv("VISION_API_COST_PER_PAGE", "0.005"))  # $5 per 1,000 calls
        self.llm_input_cost_per_m_tokens = float(os.getenv("LLM_INPUT_COST_PER_M", "0.15"))
        self.llm_output_cost_per_m_tokens = float(os.getenv("LLM_OUTPUT_COST_PER_M", "0.60"))
        self.max_cost_target_per_1k_pages = float(os.getenv("MAX_COST_TARGET_PER_1K", "10.00"))  # $10 target

        self.model_calls: Dict[str, int] = {
            "native_extractor": 0,
            "local_ocr": 0,
            "external_vision_api": 0,
            "external_llm_api": 0
        }
        self.total_estimated_cost: float = 0.0
        self.total_pages_processed: int = 0

    def record_call(self, provider_type: str, pages: int = 1, tokens: int = 0):
        """Records an extraction operation and computes incremental cost."""
        self.total_pages_processed += pages
        if provider_type in self.model_calls:
            self.model_calls[provider_type] += 1
        else:
            self.model_calls[provider_type] = 1

        if provider_type == "external_vision_api":
            self.total_estimated_cost += (pages * self.vision_api_cost_per_page)
        elif provider_type == "external_llm_api":
            self.total_estimated_cost += (tokens / 1_000_000.0) * self.llm_input_cost_per_m_tokens

    def get_cost_per_1000_pages(self, pages_processed: int, current_cost: float) -> float:
        """Calculates projected cost per 1,000 pages."""
        if pages_processed == 0:
            return 0.0
        return round((current_cost / pages_processed) * 1000.0, 4)

    def is_within_budget(self, pages_processed: int, current_cost: float) -> bool:
        """Checks if current projection is under $10 per 1,000 pages."""
        cost_per_1k = self.get_cost_per_1000_pages(pages_processed, current_cost)
        return cost_per_1k <= self.max_cost_target_per_1k_pages

    def get_metrics_summary(self) -> Dict[str, Any]:
        cost_per_1k = self.get_cost_per_1000_pages(self.total_pages_processed, self.total_estimated_cost)
        return {
            "total_pages_processed": self.total_pages_processed,
            "total_estimated_cost_usd": round(self.total_estimated_cost, 4),
            "estimated_cost_per_1000_pages_usd": cost_per_1k,
            "max_cost_target_per_1k_pages_usd": self.max_cost_target_per_1k_pages,
            "is_within_budget": cost_per_1k <= self.max_cost_target_per_1k_pages,
            "model_calls": dict(self.model_calls)
        }
