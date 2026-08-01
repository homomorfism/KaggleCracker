"""One adapter for OpenAI and DeepSeek.

DeepSeek's API is deliberately OpenAI-shaped, so the two differ only by base
URL, model name, price — and one parameter name. Newer OpenAI models reject
`max_tokens` and require `max_completion_tokens`; DeepSeek takes `max_tokens`.
That single difference is a field on the adapter rather than a second class.
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

# Newer OpenAI models reject `max_tokens` outright; DeepSeek only knows the old
# name. Sending the wrong one is a 400, not a silent fallback.
MAX_TOKENS_PARAM = {
    "openai": "max_completion_tokens",
    "deepseek": "max_tokens",
}

# finish_reason values that mean the response was cut off at the ceiling.
_TRUNCATED_REASONS = {"length", "max_tokens"}


class OpenAICompatibleRuntime:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        if settings.provider not in MAX_TOKENS_PARAM:
            raise ValueError(
                f"{type(self).__name__} serves {sorted(MAX_TOKENS_PARAM)}, "
                f"not '{settings.provider}'"
            )
        self.settings = settings
        self.provider = settings.provider
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=settings.max_retries,
            timeout=settings.request_timeout_s,
        )

    def generate(self, prompt: str, *, action: Action = "draft") -> GenerateResult:
        model = self.settings.model_for(action)
        kwargs: dict = {MAX_TOKENS_PARAM[self.provider]: self.settings.max_tokens}

        # Both providers spell this the same way. Omitted entirely when None,
        # because non-reasoning models reject the parameter rather than ignore it.
        if self.settings.reasoning_effort:
            kwargs["reasoning_effort"] = self.settings.reasoning_effort

        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        choice = resp.choices[0]
        text = choice.message.content or ""
        finish = choice.finish_reason or ""
        status, code, error_text = classify_output(
            text, hit_token_ceiling=finish in _TRUNCATED_REASONS
        )
        usage = _normalize_usage(resp.usage)

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
            stop_reason=finish,
        )


def _normalize_usage(usage) -> TokenUsage:
    """Both providers report cached input, under different names.

    DeepSeek exposes `prompt_cache_hit_tokens`; OpenAI nests it under
    `prompt_tokens_details.cached_tokens`. Cached input is billed at a small
    fraction of fresh input (on DeepSeek, ~1/50th), so missing it would make
    long-running trees look far more expensive than they are.
    """
    if usage is None:
        return TokenUsage()

    cached = getattr(usage, "prompt_cache_hit_tokens", None)
    if cached is None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0

    return TokenUsage(
        tokens_in=usage.prompt_tokens or 0,
        tokens_out=usage.completion_tokens or 0,
        cached_in=cached or 0,
    )


__all__ = ["OpenAICompatibleRuntime", "GenerateStatus"]
