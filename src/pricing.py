"""List prices used to *estimate* the cost of a run.

The numbers are Anthropic first-party API list prices in USD per million tokens,
as published on the date in ``PRICING_DATE``. They are an estimate: volume
discounts, batch pricing, long-context tiers and price changes are not modelled.
Every cost figure the logger writes carries this date so it can be re-priced.

To use a different table (a newer rate card, partner pricing), point the
``PRICING_JSON`` environment variable at a JSON file with the same shape as
``DEFAULT_PRICING``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

PRICING_DATE = "2026-10-06"
PRICING_SOURCE = "Anthropic API list prices (platform.claude.com/pricing), cached on PRICING_DATE"


class ModelPrice(TypedDict, total=False):
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens
    cache_read: float  # USD per 1M cached input tokens read (defaults to 0.1 x input)
    cache_write: float  # USD per 1M cache-creation input tokens (defaults to 1.25 x input)


DEFAULT_PRICING: dict[str, ModelPrice] = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25},
    "claude-fable-5": {"input": 10.0, "output": 50.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    # Haiku 5.5 list price is for prompts up to 100K tokens; longer prompts cost more.
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


def load_pricing() -> dict[str, ModelPrice]:
    """The default table, or the one named by ``PRICING_JSON`` if set."""
    override = os.environ.get("PRICING_JSON")
    if not override:
        return DEFAULT_PRICING
    data = json.loads(Path(override).read_text(encoding="utf-8"))
    return {str(k): v for k, v in data.items()}


@dataclass
class CostEstimate:
    model: str
    priced: bool
    pricing_date: str
    input_usd: float = 0.0
    output_usd: float = 0.0
    cache_read_usd: float = 0.0
    cache_write_usd: float = 0.0
    total_usd: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    pricing: dict[str, ModelPrice] | None = None,
) -> CostEstimate:
    """Estimate the USD cost of one run from its token counts.

    A model missing from the table yields ``priced=False`` and zero cost, never a guess.
    """
    table = pricing if pricing is not None else load_pricing()
    price = table.get(model)
    if price is None:
        return CostEstimate(
            model=model, priced=False, pricing_date=PRICING_DATE,
            note=f"no list price for {model!r} in the pricing table; add it or set PRICING_JSON",
        )
    per_m = 1_000_000
    in_rate = float(price["input"])
    out_rate = float(price["output"])
    read_rate = float(price.get("cache_read", 0.1 * in_rate))
    write_rate = float(price.get("cache_write", 1.25 * in_rate))
    est = CostEstimate(
        model=model,
        priced=True,
        pricing_date=PRICING_DATE,
        input_usd=input_tokens * in_rate / per_m,
        output_usd=output_tokens * out_rate / per_m,
        cache_read_usd=cache_read_input_tokens * read_rate / per_m,
        cache_write_usd=cache_creation_input_tokens * write_rate / per_m,
        note=f"list prices as of {PRICING_DATE}; estimate only",
    )
    est.total_usd = est.input_usd + est.output_usd + est.cache_read_usd + est.cache_write_usd
    for name in ("input_usd", "output_usd", "cache_read_usd", "cache_write_usd", "total_usd"):
        setattr(est, name, round(getattr(est, name), 6))
    return est
