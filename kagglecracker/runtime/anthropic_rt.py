"""Native Anthropic adapter.

Genuinely different from the OpenAI shape, which is why it gets its own class
rather than riding the compatibility endpoint: the system prompt sits outside
the messages array, `max_tokens` is required rather than optional, usage comes
back as `input_tokens`/`output_tokens`, and a request can come back declined
with a normal HTTP 200.

That last one matters. A refusal is `stop_reason == "refusal"` with empty or
partial content — code that reads `content[0].text` unconditionally raises on
it, so `stop_reason` is checked before the content is touched.
"""

from __future__ import annotations

from kagglecracker.config import Settings
from kagglecracker.runtime.base import (
    Action,
    GenerateResult,
    GenerateStatus,
    TokenUsage,
    classify_output,
)
from kagglecracker.runtime.pricing import estimate_usd


class AnthropicRuntime:
    def __init__(self, settings: Settings) -> None:
        import anthropic

        if settings.provider != "anthropic":
            raise ValueError(
                f"{type(self).__name__} serves 'anthropic', not '{settings.provider}'"
            )
        self.settings = settings
        self.provider = "anthropic"
        self.client = anthropic.Anthropic(
            api_key=settings.api_key,
            max_retries=settings.max_retries,
            timeout=settings.request_timeout_s,
        )

    def generate(self, prompt: str, *, action: Action = "draft") -> GenerateResult:
        model = self.settings.model_for(action)

        kwargs: dict = {}
        # Anthropic spells reasoning depth `output_config.effort`, not
        # `reasoning_effort`. Same concept, different wire name — normalizing it
        # here is exactly why this is a separate adapter.
        if self.settings.reasoning_effort:
            kwargs["output_config"] = {"effort": self.settings.reasoning_effort}

        resp = self.client.messages.create(
            model=model,
            # Required here, unlike the OpenAI shape where it is optional. It
            # also caps thinking as well as visible text on current models, so
            # a tight value truncates a perfectly good answer.
            max_tokens=self.settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        usage = TokenUsage(
            tokens_in=resp.usage.input_tokens or 0,
            tokens_out=resp.usage.output_tokens or 0,
            cached_in=(resp.usage.cache_read_input_tokens or 0),
        )
        stop_reason = resp.stop_reason or ""

        # Check stop_reason before reading content — a refusal can carry no
        # content blocks at all.
        if stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None)
            return GenerateResult(
                status=GenerateStatus.REFUSAL,
                code=None,
                raw_text="",
                usage=usage,
                provider=self.provider,
                model=model,
                usd=estimate_usd(
                    self.provider, model, usage.tokens_in, usage.tokens_out, usage.cached_in
                ),
                error_text=(
                    f"The provider declined this request"
                    f"{f' (category: {category})' if category else ''}. "
                    "Not a code defect — the node was never generated."
                ),
                prompt_chars=len(prompt),
                stop_reason=stop_reason,
            )

        text = "".join(b.text for b in resp.content if b.type == "text")
        status, code, error_text = classify_output(
            text, hit_token_ceiling=stop_reason == "max_tokens"
        )

        return GenerateResult(
            status=status,
            code=code,
            raw_text=text,
            usage=usage,
            provider=self.provider,
            model=model,
            usd=estimate_usd(
                self.provider, model, usage.tokens_in, usage.tokens_out, usage.cached_in
            ),
            error_text=error_text,
            prompt_chars=len(prompt),
            stop_reason=stop_reason,
        )


__all__ = ["AnthropicRuntime"]
