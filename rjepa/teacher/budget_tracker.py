"""
Budget tracker for teacher API usage.

Tracks token usage and estimated costs across multiple teacher LLMs.
"""
import logging
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# Pricing per 1M tokens (as of 2025, approximate)
PRICING = {
    # Claude
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},

    # GPT
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},

    # Default fallback
    "default": {"input": 5.0, "output": 15.0},
}


class BudgetTracker:
    """
    Track API usage and costs.

    Example:
        >>> tracker = BudgetTracker(max_budget_usd=50.0)
        >>> tracker.record_usage("claude-3-5-sonnet", input_tokens=1000, output_tokens=500)
        >>> print(tracker.get_total_cost())
        0.0105
    """

    def __init__(
        self,
        max_budget_usd: float = 50.0,
        log_path: Optional[str] = "logs/teacher/budget.json"
    ):
        """
        Initialize Budget Tracker.

        Args:
            max_budget_usd: Maximum budget in USD
            log_path: Path to save budget log
        """
        self.max_budget_usd = max_budget_usd
        self.log_path = log_path

        # Usage tracking
        self.usage: Dict[str, Dict] = {}  # model -> {input_tokens, output_tokens, cost}
        self.total_cost = 0.0
        self.start_time = datetime.now()

        # Load existing log if available
        if log_path and Path(log_path).exists():
            self._load_log()

        logger.info(f"Budget tracker initialized (max_budget=${max_budget_usd})")

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """
        Record API usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        # Get pricing
        pricing = self._get_pricing(model)

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cost = input_cost + output_cost

        # Update tracking
        if model not in self.usage:
            self.usage[model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0.0,
                "requests": 0,
            }

        self.usage[model]["input_tokens"] += input_tokens
        self.usage[model]["output_tokens"] += output_tokens
        self.usage[model]["cost"] += cost
        self.usage[model]["requests"] += 1

        self.total_cost += cost

        # Check budget
        if self.total_cost > self.max_budget_usd:
            logger.warning(f"Budget exceeded: ${self.total_cost:.2f} > ${self.max_budget_usd:.2f}")

        # Save log
        if self.log_path:
            self._save_log()

        logger.debug(f"Recorded usage: {model} | input={input_tokens} output={output_tokens} cost=${cost:.4f}")

    def get_total_cost(self) -> float:
        """
        Get total cost across all models.

        Returns:
            Total cost in USD
        """
        return self.total_cost

    def get_usage_summary(self) -> Dict:
        """
        Get usage summary.

        Returns:
            Dict with usage stats per model
        """
        return {
            "models": self.usage,
            "total_cost": self.total_cost,
            "max_budget": self.max_budget_usd,
            "budget_remaining": self.max_budget_usd - self.total_cost,
            "budget_used_pct": (self.total_cost / self.max_budget_usd) * 100,
            "start_time": self.start_time.isoformat(),
            "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
        }

    def is_budget_exceeded(self) -> bool:
        """
        Check if budget is exceeded.

        Returns:
            True if budget exceeded
        """
        return self.total_cost > self.max_budget_usd

    def get_budget_remaining(self) -> float:
        """
        Get remaining budget.

        Returns:
            Remaining budget in USD
        """
        return max(0.0, self.max_budget_usd - self.total_cost)

    def reset(self):
        """Reset budget tracker."""
        self.usage = {}
        self.total_cost = 0.0
        self.start_time = datetime.now()
        logger.info("Budget tracker reset")

    def _get_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name

        Returns:
            Dict with "input" and "output" pricing (USD per 1M tokens)
        """
        # Try exact match
        if model in PRICING:
            return PRICING[model]

        # Try partial match
        for key in PRICING:
            if key in model.lower():
                return PRICING[key]

        # Fallback to default
        logger.warning(f"Unknown model pricing: {model}, using default")
        return PRICING["default"]

    def _save_log(self):
        """Save budget log to file."""
        if not self.log_path:
            return

        # Create directory if needed
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        # Save
        with open(self.log_path, 'w') as f:
            json.dump(self.get_usage_summary(), f, indent=2)

    def _load_log(self):
        """Load budget log from file."""
        try:
            with open(self.log_path, 'r') as f:
                data = json.load(f)

            self.usage = data.get("models", {})
            self.total_cost = data.get("total_cost", 0.0)
            self.max_budget_usd = data.get("max_budget", self.max_budget_usd)

            if "start_time" in data:
                self.start_time = datetime.fromisoformat(data["start_time"])

            logger.info(f"Loaded budget log: ${self.total_cost:.2f} spent")

        except Exception as e:
            logger.error(f"Failed to load budget log: {e}")

    def __repr__(self) -> str:
        return (
            f"BudgetTracker("
            f"cost=${self.total_cost:.2f}, "
            f"remaining=${self.get_budget_remaining():.2f}, "
            f"used={self.total_cost/self.max_budget_usd*100:.1f}%"
            f")"
        )
