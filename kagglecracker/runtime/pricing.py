"""Per-model token prices, and the assert that keeps `max_usd` armed.

Prices move. This table is a snapshot, sourced 2026-07-31 from each provider's
own pricing page — not from memory. Re-check it when you change models.

The failure mode this module exists to prevent is silent, not loud: an unpriced
model computes $0.00 per node, `max_usd` never trips, and an overnight run bills
against a cap that was never armed. So an unknown (provider, model) is a hard
error at startup, never a zero.

    Settings.models_in_use()  ──►  assert_all_priced()  ──►  startup fails loudly
                                          │
    GenerateResult.tokens_*   ──►  estimate_usd()  ──►  runs.usd_spent  ──► max_usd
"""

from __future__ import annotations

from dataclasses import dataclass

from kagglecracker.config import Provider

# Cached input is billed at a fraction of fresh input on all three providers.
# Where a provider publishes an explicit cache price we use it; otherwise this
# is the documented ratio (OpenAI and Anthropic both bill cache reads at ~0.1x).
DEFAULT_CACHE_READ_RATIO = 0.1


@dataclass(frozen=True, slots=True)
class Price:
    """USD per 1M tokens."""

    input_usd: float
    output_usd: float
    cached_input_usd: float | None = None

    @property
    def cache_read_usd(self) -> float:
        if self.cached_input_usd is not None:
            return self.cached_input_usd
        return self.input_usd * DEFAULT_CACHE_READ_RATIO


# Sourced 2026-07-31. Prices are USD per 1M tokens (input, output).
PRICE_PER_1M: dict[tuple[Provider, str], Price] = {
    # --- OpenAI (developers.openai.com/api/docs/pricing) --------------------
    ("openai", "gpt-5.6-sol"): Price(5.00, 30.00),
    ("openai", "gpt-5.6-terra"): Price(2.00, 12.00),
    ("openai", "gpt-5.6-luna"): Price(0.20, 1.20),
    ("openai", "gpt-5.5"): Price(5.00, 30.00),
    ("openai", "gpt-5-mini"): Price(0.25, 2.00),
    ("openai", "gpt-5-nano"): Price(0.05, 0.40),
    ("openai", "o3"): Price(2.00, 8.00),
    # --- DeepSeek (api-docs.deepseek.com/quick_start/pricing) ---------------
    # NOTE: `deepseek-chat` and `deepseek-reasoner` no longer exist. Cache-hit
    # input is ~50x cheaper than cache-miss, which is why it is explicit here.
    # DeepSeek has announced peak/off-peak pricing at 2x during peak hours —
    # if that lands, these become the off-peak floor, not the actual bill.
    ("deepseek", "deepseek-v4-flash"): Price(0.14, 0.28, cached_input_usd=0.0028),
    ("deepseek", "deepseek-v4-pro"): Price(0.435, 0.87, cached_input_usd=0.003625),
    # --- Anthropic (claude.com/pricing) -------------------------------------
    # claude-sonnet-5 is priced at its list rate, not the $2/$10 introductory
    # rate that runs through 2026-08-31. Deliberate: a budget cap should
    # overestimate, so `max_usd` trips early rather than late.
    ("anthropic", "claude-fable-5"): Price(10.00, 50.00),
    ("anthropic", "claude-opus-5"): Price(5.00, 25.00),
    ("anthropic", "claude-opus-4-8"): Price(5.00, 25.00),
    ("anthropic", "claude-sonnet-5"): Price(3.00, 15.00),
    ("anthropic", "claude-haiku-4-5"): Price(1.00, 5.00),
}


class UnpricedModelError(RuntimeError):
    """A configured model has no price entry, so `max_usd` cannot be enforced."""


def price_for(provider: Provider, model: str) -> Price:
    try:
        return PRICE_PER_1M[(provider, model)]
    except KeyError:
        known = sorted(m for p, m in PRICE_PER_1M if p == provider)
        raise UnpricedModelError(
            f"No price entry for {provider}/{model}. Every configured model must be "
            f"priced or max_usd silently computes $0.00 and never trips.\n"
            f"Priced {provider} models: {', '.join(known) or '(none)'}\n"
            f"Add the model to PRICE_PER_1M in kagglecracker/runtime/pricing.py, "
            f"reading the current rate from the provider's pricing page."
        ) from None


def assert_all_priced(provider: Provider, models: set[str]) -> None:
    """Called at startup. Raises on the first unpriced model."""
    for model in sorted(models):
        price_for(provider, model)


def estimate_usd(
    provider: Provider,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cached_tokens_in: int = 0,
) -> float:
    """Cost of one call. `cached_tokens_in` is a subset of `tokens_in`."""
    price = price_for(provider, model)
    fresh_in = max(0, tokens_in - cached_tokens_in)
    return (
        fresh_in * price.input_usd
        + cached_tokens_in * price.cache_read_usd
        + tokens_out * price.output_usd
    ) / 1_000_000
